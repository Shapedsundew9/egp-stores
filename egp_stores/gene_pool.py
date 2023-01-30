"""Gene pool management for Erasmus GP."""

from copy import deepcopy
from json import load
from logging import DEBUG, Logger, NullHandler, getLogger
from os.path import dirname, join
from typing import Any, Literal
from functools import partial 
from itertools import count

from egp_population.typing import PopulationsNorm
from egp_types.conversions import compress_json, decompress_json, memoryview_to_bytes
from egp_types.gc_graph import gc_graph
from egp_types.gc_type_tools import NUM_PGC_LAYERS
from egp_types.reference import ref_from_sig, ref_str
from egp_types.xgc_validator import gGC_entry_validator
from egp_types.reference import reference
from numpy import histogram
from numpy.core.fromnumeric import mean
from pypgtable import table
from pypgtable.typing import Conversions, PtrMap, TableConfig, TableConfigNorm, TableSchema
from pypgtable.validators import raw_table_config_validator
from .typing import GenePoolConfig, GenePoolConfigNorm
from egp_utils.common import merge

from .gene_pool_cache import GPC_HIGHER_LAYER_COLS, GPC_UPDATE_RETURNING_COLS, gene_pool_cache, xGC
from .gene_pool_common import GP_HIGHER_LAYER_COLS, GP_RAW_TABLE_SCHEMA, GP_UPDATE_RETURNING_COLS
from .genetic_material_store import genetic_material_store
from .genomic_library import GL_HIGHER_LAYER_COLS, GL_RAW_TABLE_SCHEMA, GL_SIGNATURE_COLUMNS, UPDATE_STR, genomic_library, sql_functions

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
# TODO: Add a _LOG_CONSISTENCY which additionally does consistency checking
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


def compress_igraph(obj: gc_graph) -> bytes | memoryview | bytearray | None:
    """Extract the internal representation and compress."""
    return compress_json(obj.save())


def decompress_igraph(obj: bytes) -> gc_graph:
    """Create a gc_graph() from an decompressed internal representation."""
    return gc_graph(internal=decompress_json(obj))


_GP_CONVERSIONS: Conversions = (
    ('graph', compress_json, decompress_json),
    ('meta_data', compress_json, decompress_json),  # TODO: Why store this?
    ('inputs', None, memoryview_to_bytes),
    ('outputs', None, memoryview_to_bytes),
    ('igraph', compress_igraph, decompress_igraph)
)


# Tree structure
_LEL: Literal['gca_ref'] = 'gca_ref'
_REL: Literal['gcb_ref'] = 'gcb_ref'
_PEL: Literal['pgc_ref'] = 'pgc_ref'
_NL: Literal['ref'] = 'ref'
_PTR_MAP: PtrMap = {
    _LEL: _NL,
    _REL: _NL
}
_PTR_MAP_PLUS_PGC: PtrMap = {
    _LEL: _NL,
    _REL: _NL,
    _PEL: _NL
}

_OWNER_ID_UPDATE_SQL: str = "{owner_id} = (({owner_id}::BIGINT + 1::BIGINT) & x'7FFFFFFF::BIGINT)::INTEGER')"

with open(join(dirname(__file__), "formats/gp_metrics_table_format.json"), "r", encoding="utf8") as file_ptr:
    _GP_METRICS_TABLE_SCHEMA: TableSchema = load(file_ptr)
with open(join(dirname(__file__), "formats/gp_pgc_metrics_table_format.json"), "r", encoding="utf8") as file_ptr:
    _GP_PGC_METRICS_TABLE_SCHEMA: TableSchema = load(file_ptr)
with open(join(dirname(__file__), "formats/gp_meta_table_format.json"), "r", encoding="utf8") as file_ptr:
    _GP_META_TABLE_SCHEMA: TableSchema = load(file_ptr)


# GC queries
_LOAD_GL_COLUMNS: tuple[str, ...] = tuple(k for k in GL_RAW_TABLE_SCHEMA.keys() if k not in GL_HIGHER_LAYER_COLS)
_LOAD_GP_COLUMNS: tuple[str, ...] = tuple(k for k in GP_RAW_TABLE_SCHEMA.keys() if k not in GP_HIGHER_LAYER_COLS)
_REF_SQL: str = 'WHERE ({ref} = ANY({matches}))'
_SIGNATURE_SQL: str = 'WHERE ({signature} = ANY({matches}))'
_LOAD_GPC_SQL: str = ('WHERE {population_uid} = {puid} ORDER BY {survivability} DESC LIMIT {limit}')
_LOAD_PGC_SQL: str = ('WHERE {pgc_f_count}[{layer}] > 0 AND NOT({ref} = ANY({exclusions})) '
                      'ORDER BY {pgc_fitness}[{layer}] DESC LIMIT {limit}')
_LOAD_CODONS_SQL: str = "WHERE {creator}::text = '22c23596-df90-4b87-88a4-9409a0ea764f'"


# The gene pool default config
_DEFAULT_GP_CONFIG: TableConfigNorm = raw_table_config_validator.normalized({
    'database': {
        'dbname': 'erasmus_db'
    },
    'table': 'gene_pool',
    'ptr_map': _PTR_MAP,
    'schema': GP_RAW_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
    'conversions': _GP_CONVERSIONS
})
_DEFAULT_GP_METRICS_CONFIG: TableConfigNorm = raw_table_config_validator.normalized({
    'database': {
        'dbname': 'erasmus_db'
    },
    'table': 'gp_metrics',
    'schema': _GP_METRICS_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
})
_DEFAULT_PGC_METRICS_CONFIG: TableConfigNorm = raw_table_config_validator.normalized({
    'database': {
        'dbname': 'erasmus_db'
    },
    'table': 'pgc_metrics',
    'schema': _GP_PGC_METRICS_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
})
_DEFAULT_GP_META_CONFIG: TableConfigNorm = raw_table_config_validator.normalized({
    'database': {
        'dbname': 'erasmus_db'
    },
    'table': 'owners',
    'schema': _GP_META_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
})
_DEFAULT_CONFIGS: GenePoolConfigNorm = {
    'gene_pool': _DEFAULT_GP_CONFIG,
    'meta_data': _DEFAULT_GP_META_CONFIG,
    'gp_metrics': _DEFAULT_GP_METRICS_CONFIG,
    'pgc_metrics': _DEFAULT_PGC_METRICS_CONFIG,
}


def default_config() -> GenePoolConfigNorm:
    """Return a deepcopy of the default genomic library configuration.

    The copy may be modified and used to create a genomic library instance.

    Returns
    -------
    The default genomic_library() configuration.
    """
    return deepcopy(_DEFAULT_CONFIGS)


class gene_pool(genetic_material_store):
    """Store of transient genetic codes & associated data for a population.

    The gene_pool is responsible for:
        1. Populating calculable entry fields.
        2. Providing an application interface to the fields.
        3. Persistence of updates to the gene pool.
        4. Local caching of GC's.

    The gene pool must be consistent i.e. no entry can depend on a genetic code
    that is not in the gene_pool.

    The primary difference with the genomic_library is the presence of transient data
    and the fast (and space saving) UID method of referencing and storing active GC's.
    For the same reasons validation is not performed unless in debug mode.

    The gene_pool is more local (faster to access) than the genomic library
    due to the heavy transaction load. It is also designed to be multi-process
    memory efficient.

    The public member self.pool is the local cache of gGC's. Querying or writing to
    the GP should only happen during an egp_physics.physics.proximity_search() or
    when filling / flushing the GPC.
    """
    # TODO: Default genomic_library should be local host

    def __init__(self, populations: PopulationsNorm, glib: genomic_library | None = None,
                 config: GenePoolConfig | GenePoolConfigNorm = {}) -> None:
        """Connect to or create a gene pool.

        The gene pool data persists in a postgresql database. Multiple
        instances of the gene_pool() class can connect to the same database
        (use the same configuration).

        Args
        ----
        populations: The populations being worked on by this worker.
        genomic_library: Source of genetic material.
        configs: The config is deep copied by pypgtable.
        """
        # pool is a python2 type dictionary-like object of ref:gGC
        # All gGC's in pool must be valid & any modified are sync'd to the gene pool table
        # in the database at the end of the target epoch.
        def normalize(k, v) -> dict[Any, Any]:
            return merge(v, raw_table_config_validator.normalized(config.get(k, {})), no_new_keys=True)
        self.config: GenePoolConfigNorm = {k: normalize(k, v) for k, v in _DEFAULT_CONFIGS.items()}  # type: ignore

        self._gl: genomic_library = genomic_library() if glib is None else glib

        self.pool: gene_pool_cache = gene_pool_cache()
        _logger.info('Gene Pool Cache created.')
        self._pool: table = table(self.config['gene_pool'])
        _logger.info('Established connection to Gene Pool table.')

        self._tables: dict[str, table] = {k: table(v) for k, v in self.config.items()}
        _logger.info('Established connections to gene pool tables.')

        # Modify the update strings to use the right table for the gene pool.
        self._update_str: str = UPDATE_STR.replace('__table__', self.config['gene_pool']['table'])

        # If this instance created the gene pool then it is responsible for configuring
        # setting up database functions and initial population.
        if self._pool.raw.creator:
            # FIXME: How are other workers held back until this is executed?
            self._pool.raw.arbitrary_sql(sql_functions(), read=False)

        self.owner_id: int | None = None
        self.next_reference: partial[int] | None = None

        # Fill the local cache with populations in the persistent GP.
        self._populate_local_cache()

    def sub_process_init(self) -> None:
        """Define subprocess specific values."""
        self.owner_id = next(self._tables['meta_data'].update(_OWNER_ID_UPDATE_SQL, returning=('last_owner_id',), container='tuple'))[0]
        self.next_reference = partial(reference, owner_id=self.owner_id, counter=count())

    def creator(self) -> bool:
        """True if this process created the gene pool table in the database."""
        return self._pool.raw.creator

    def _populate_local_cache(self) -> None:
        """Gather the latest and greatest from the GP.

        1. Load all the codons.
        2. For each population select the population size with the highest survivability.
        3. Pull in the pGC's that created all the selected population & sub-gGCs
        4. Pull in the best pGC's at each layer.
        5. If not enough quality population pull from higher layer or create eGC's.
        """
        self.pull(list(self._gl.select(_LOAD_CODONS_SQL, columns=('signature',), container='tuple')))
        for population in self._populations.values():
            literals: dict[str, Any] = {'limit': population['size'], 'puid': population['uid']}
            for ggc in self._pool.select(_LOAD_GPC_SQL, literals, _LOAD_GP_COLUMNS):
                self.pool[ggc['ref']] = ggc

            self._pool.raw.ptr_map_def(_PTR_MAP_PLUS_PGC)
            for ggc in self._pool.recursive_select(_REF_SQL, {'matches': list(self.pool.keys())}):
                self.pool[ggc['ref']] = ggc

            self._pool.raw.ptr_map_def(_PTR_MAP)
            literals = {'limit': population['size']}
            for layer in range(NUM_PGC_LAYERS):
                literals['exclusions'] = self.pool.pgc_refs()
                literals['layer'] = layer
                for pgc in self._pool.select(_LOAD_PGC_SQL, literals):
                    self.pool[pgc['ref']] = pgc

    def _ref_from_sig(self, sig: bytes | None) -> int | None:
        """Convert a signature to a reference handling clashes.

        Args
        ----
        sig: Genomic Library signature

        Returns
        -------
        64 bit reference. See reference() for bitfields.
        """
        if sig is None:
            return None
        ref: int = ref_from_sig(sig)
        if ref in self.pool and self.pool[ref]['signature'] != sig:
            shift: int = 0
            while ref in self.pool and self.pool[ref]['signature'] != sig:
                _logger.warning(f"Hashing clash at shift {shift} for {ref_str(ref)} and {sig.hex()}.")
                # FIXME: Shift can be used if we can do an atomic update of the GP database
                assert not shift
                ref = ref_from_sig(sig, shift := shift + 1)
                assert shift < 193, "193 consecutive hashing clashes. Hmmmm!"
        return ref

    def pull(self, signatures: list[bytes], population_uid: int | None = None) -> None:
        """Pull aGCs and all sub-GC's recursively from the genomic library to the gene pool.

        LGC's are converted to gGC's.
        Higher layer fields are updated.
        Nodes & edges are added the the GP graph.
        SHA256 signatures are replaced by GP references.

        NOTE: This *MUST* be the only function pulling GC's into the GP from the GL.

        Args
        ----
        signatures: Signatures to pull from the genomic library.
        population_uid: Population UID to label ALL top level GC's with
        """
        if _LOG_DEBUG:
            _logger.debug(f'Recursively pulling {signatures} into Gene Pool for population {population_uid}.')
        sig_ref_map: dict[bytes, int] = {}
        for ggc in self._gl.recursive_select(_SIGNATURE_SQL, {'matches': signatures}, _LOAD_GL_COLUMNS):
            if population_uid is not None and ggc['signature'] in signatures:
                ggc['population_uid'] = population_uid

            # Map signatures to references
            sig_ref_map.update({ggc[field]: self._ref_from_sig(ggc[field]) for field in GL_SIGNATURE_COLUMNS})
            ggc.update({field + '_ref': sig_ref_map[ggc[field]] for field in GL_SIGNATURE_COLUMNS if field != 'signature'})
            ggc['ref'] = sig_ref_map[ggc['signature']]

            # Set the higher layer fields
            # A higher layer field starts with an underscore '_' and has an underscoreless counterpart.
            # e.g. '_field' and 'field'. The _field holds the value of field when the GC was pulled from
            # the higher layer. i.e. after being pulled from the GMS _field must = field. field can then
            # be modified by the lower layer. NB: Updating the value back into the GMS is a bit more complex.
            ggc.update({k: ggc[k[1:]] for k in GP_HIGHER_LAYER_COLS})

            # Push to GP
            # FIXME: Do this update in batch for better performance.
            updates = next(self._pool.upsert((ggc,), self._update_str, {}, GP_UPDATE_RETURNING_COLS))
            ggc.update(updates)

            # Push to GPC
            ggc.update({k: ggc[k[1:]] for k in GPC_HIGHER_LAYER_COLS})
            self.pool[ggc['ref']] = ggc
            if _LOG_DEBUG:
                if not gGC_entry_validator.validate(ggc):
                    _logger.error(ggc)
                    _logger.error(gGC_entry_validator.error_str())
                    assert False, "GGC is not valid!"

    def push(self) -> None:
        """Insert or update locally modified gGC's into the persistent gene_pool.

        NOTE: This *MUST* be the only function pushing GC's to the persistent GP.
        """
        # To keep pylance happy about 'possibly unbound'
        updated_gcs: list[xGC] = []
        modified_gcs: list[xGC] = updated_gcs
        if _LOG_DEBUG:
            _logger.debug('Validating GP DB entries.')
            updated_gcs = []
            modified_gcs: list[xGC] = list(self.pool.modified(all_fields=True))
            for ggc in modified_gcs:
                if not gGC_entry_validator.validate(ggc):
                    _logger.error(f'Modified gGC invalid:\n{gGC_entry_validator.error_str()}.')
                    raise ValueError('Modified gGC is invalid. See log.')

        for updated_gc in self._pool.upsert(self.pool.modified(), self._update_str, {}, GPC_UPDATE_RETURNING_COLS):
            ggc: xGC = self.pool[updated_gc['ref']]
            ggc['__modified__'] = False
            ggc.update(updated_gc)
            ggc.update({k: ggc[k[1:]] for k in GPC_HIGHER_LAYER_COLS})
            if _LOG_DEBUG:
                updated_gcs.append(ggc)

        if _LOG_DEBUG:
            modified_in_gpc: tuple[str, ...] = tuple((ref_str(v['ref']) for v in self.pool.modified()))
            assert not modified_gcs, f"Modified gGCs were not written to the GP: {modified_in_gpc}"
            assert len(modified_gcs) == len(updated_gcs), "The number of updated and modified gGCs differ!"

            # Check updates are correct
            for uggc, mggc in zip(updated_gcs, modified_gcs):
                if not gGC_entry_validator.validate(uggc):
                    _logger.error(f'Updated gGC invalid:\n{gGC_entry_validator.error_str()}.')
                    raise ValueError('Updated gGC is invalid. See log.')

                # Sanity the read-only fields
                for field in filter(lambda x: mggc.fields[x].get('read_only', False), mggc.fields):  # pylint: disable=cell-var-from-loop
                    assert uggc[field] == mggc[field], "Read-only fields must remain unchanged!"

                # Writable fields may not have changed. Only one we can be sure of is 'updated'.
                assert uggc['updated'] > mggc['updated'], "Updated timestamp must be newer!"

    def metrics(self) -> None:
        """Calculate and record metrics for the GP.

        The data is based on the GPC and is calculated as a delta since the last call.
        """
        self.pgc_metrics()
        self.gp_metrics()

    def pgc_metrics(self) -> None:
        """Per pGC layer metrics."""
        # FIXME: This could be way more memory and CPU efficient.
        pgcs = [self.pool[ref] for ref in self.pool.pgc_refs()]
        for layer, evolutions in filter(lambda x: x[1], enumerate(self.layer_evolutions)):
            lpgcs = tuple(gc for gc in pgcs if gc['pgc_f_count'][layer])
            fitness = [individual['pgc_fitness'][layer] for individual in lpgcs]
            evolvability = [individual['pgc_evolvability'][layer] for individual in lpgcs]
            generation = [individual['generation'] for individual in lpgcs]
            gc_count = [individual['num_codes'] for individual in lpgcs]
            c_count = [individual['num_codons'] for individual in lpgcs]
            self._metrics['pgc_metrics'].insert([{
                'layer': layer,
                'count': len(fitness),
                'f_max': max(fitness),
                'f_mean': mean(fitness),
                'f_dist': [int(b) for b in histogram(fitness, 10, (0, 1.0))[0]],
                'f_min': min(fitness),
                'e_max': max(evolvability),
                'e_mean': mean(evolvability),
                'e_min': min(evolvability),
                'g_max': max(generation),
                'g_mean': mean(generation),
                'g_min': min(generation),
                'gcc_max': max(gc_count),
                'gcc_mean': mean(gc_count),
                'gcc_min': min(gc_count),
                'cc_max': max(c_count),
                'cc_mean': mean(c_count),
                'cc_min': min(c_count),
                'evolutions': evolutions,
                'eps': evolutions,  # FIXME: Needs to be per second
                'performance': 0.0,
                'tag': 0,
                'worker_id': 0}])

    def gp_metrics(self) -> None:
        """Per pGC layer metrics."""
