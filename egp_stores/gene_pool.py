"""Gene pool management for Erasmus GP."""

from copy import deepcopy
from functools import partial
from os.path import dirname, join
from logging import DEBUG, INFO, WARN, ERROR, FATAL, NullHandler, getLogger
from json import load, loads, dumps
from random import random
from numpy.core.fromnumeric import mean
from numpy.random import choice
from numpy import array, float32, sum, count_nonzero, where, finfo, histogram, isfinite, concatenate, argsort, logical_and
from .genetic_material_store import genetic_material_store
from .gene_pool_table_schema import GP_TABLE_SCHEMA
from egp_types.conversions import compress_json, decompress_json, memoryview_to_bytes
from .genomic_library import UPDATE_STR, UPDATE_RETURNING_COLS, sql_functions, HIGHER_LAYER_COLS, _GL_TABLE_SCHEMA
from egp_types.ep_type import asstr, vtype, asint, interface_definition, ordered_interface_hash
from egp_types.gc_type_tools import NUM_PGC_LAYERS, is_pgc, ref_str, reference
from egp_types.eGC import eGC
from egp_physics.physics import stablize, population_GC_evolvability, pGC_fitness, select_pGC, RANDOM_PGC_SIGNATURE
from egp_physics.physics import population_GC_inherit
from egp_types.gc_graph import gc_graph
from egp_execution.execution import set_gms, create_callable
from .gene_pool_cache import gene_pool_cache, xGC
from time import time, sleep
from pypgtable import table
from functools import partial
from uuid import uuid4
from typing import Dict, List, Tuple, Any, Callable, Iterable
from itertools import count

_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
# TODO: Add a _LOG_CONSISTENCY which additionally does consistency checking
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)


_UPDATE_RETURNING_COLS = tuple((c for c in filter(lambda x: x != 'signature', UPDATE_RETURNING_COLS))) + ('ref',)
_POPULATION_IN_DEFINITION = -1
_MAX_INITIAL_LIST = 100000
_MIN_PGC_LAYER_SIZE = 100
_MAX_PGC_LAYER_SIZE = 10000
_MINIMUM_SUBPROCESS_TIME = 60
_MINIMUM_AVAILABLE_MEMORY = 128 * 1024 * 1024
_MAX_POPULATION_SIZE = 100000


def compress_igraph(x):
    """Extract the internal representation and compress."""
    return compress_json(x.save())


def decompress_igraph(x):
    """Create a gc_graph() from an decompressed internal representation."""
    return gc_graph(internal=decompress_json(x))


_GP_CONVERSIONS = (
    ('graph', compress_json, decompress_json),
    ('meta_data', compress_json, decompress_json),  # TODO: Why store this?
    ('inputs', None, memoryview_to_bytes),
    ('outputs', None, memoryview_to_bytes),
    ('igraph', compress_igraph, decompress_igraph)
)
_POPULATIONS_CONVERSIONS = (
    ('inputs', dumps, loads),
    ('outputs', dumps, loads)
)


# Tree structure
_LEL = 'gca_ref'
_REL = 'gcb_ref'
_PEL = 'pgc_ref'
_NL = 'ref'
_PTR_MAP = {
    _LEL: _NL,
    _REL: _NL,
    _PEL: _NL
}

with open(join(dirname(__file__), "formats/gp_population_table_format.json"), "r") as file_ptr:
    _GP_POPULATION_TABLE_SCHEMA = load(file_ptr)
with open(join(dirname(__file__), "formats/gp_metrics_table_format.json"), "r") as file_ptr:
    _GP_METRICS_TABLE_SCHEMA = load(file_ptr)
with open(join(dirname(__file__), "formats/population_metrics_table_format.json"), "r") as file_ptr:
    _GP_POPULATION_METRICS_TABLE_SCHEMA = load(file_ptr)
with open(join(dirname(__file__), "formats/gp_pgc_metrics_format.json"), "r") as file_ptr:
    _GP_PGC_METRICS_TABLE_SCHEMA = load(file_ptr)


# GC queries
_GL_EXCLUDE_COLUMNS = ('ancestor_a', 'ancestor_b', 'creator')
_GL_COLUMNS = tuple(k for k in _GL_TABLE_SCHEMA.keys() if k not in _GL_EXCLUDE_COLUMNS)
_LAST_OWNER_ID_QUERY_SQL = 'WHERE uid = {uid}'
_LAST_OWNER_ID_UPDATE_SQL = "{last_owner_id} = ({last_owner_id} + 1) & x'FFFFFFFF')"
_REF_SQL = 'WHERE ({ref} = ANY({matches}))'
_SIGNATURE_SQL = 'WHERE ({signature} = ANY({matches}))'
_LOAD_GPC_SQL = ('WHERE {population} = {population_uid} ORDER BY {survivability} DESC LIMIT {limit}')
_LOAD_PGC_SQL = ('WHERE {pgc_f_count}[{layer}] > 0 AND NOT({ref} = ANY({exclusions})) '
                 'ORDER BY {pgc_fitness}[{layer}] DESC LIMIT {limit}')
_LOAD_CODONS_SQL = "WHERE {creator}::text = '22c23596-df90-4b87-88a4-9409a0ea764f':UUID"


# The gene pool default config
_DEFAULT_GP_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'gene_pool',
    'ptr_map': _PTR_MAP,
    'schema': GP_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
    'conversions': _GP_CONVERSIONS
}
_DEFAULT_POPULATIONS_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'gene_pool_populations',
    'schema': _GP_POPULATION_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
    'conversions': _POPULATIONS_CONVERSIONS
}
_DEFAULT_GP_METRICS_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'gp_metrics',
    'schema': _GP_METRICS_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
}
_DEFAULT_POPULATION_METRICS_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'population_metrics',
    'schema': _GP_POPULATION_METRICS_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
}
_DEFAULT_PGC_METRICS_CONFIG = {
    'database': {
        'dbname': 'erasmus'
    },
    'table': 'pgc_metrics',
    'schema': _GP_PGC_METRICS_TABLE_SCHEMA,
    'create_table': True,
    'create_db': True,
}
_DEFAULT_CONFIGS = {
    "gp": _DEFAULT_GP_CONFIG,
    "populations": _DEFAULT_POPULATIONS_CONFIG,
    "gp_metrics": _DEFAULT_GP_METRICS_CONFIG,
    "population_metrics": _DEFAULT_POPULATION_METRICS_CONFIG,
    "pgc_metrics": _DEFAULT_PGC_METRICS_CONFIG,
}


def reference(owner:int, counters:Dict[int, count]) -> int:
    """Create a unique reference.

    It is assumed that after the 2**32th owner the 1st owner ID
    will be available again without risk to uniqueness.

    It is also assumed limiting the 

    Args
    ----
    owner: 32 bit unsigned integer uniquely identifying the counter to be used.

    Returns
    -------
    Signed 64 bit integer reference.
    """
    if owner not in counters:
        counters[owner] = count(2**32)
    return (counters[owner] + (owner << 32)) & 0xFFFFFFFFFFFFFFFF


def default_config() -> Dict[str, Dict[str, Any]]:
    """Return a deepcopy of the default genomic library configuration.

    The copy may be modified and used to create a genomic library instance.

    Returns
    -------
    (dict): The default genomic_library() configuration.
    """
    return deepcopy(_DEFAULT_CONFIGS)


def _characterization_wrapper(gc:xGC, characterize:Callable) -> Tuple[float, float]:
    """Wrap characterize() to manage out of bounds results.
    
    Args
    ----
    characterize: The characterization for the population

    Returns
    -------
    tuple(float, float): Fitness, Survivability both in the range 0.0 <= x <= 1.0 or None   
    """
    values = characterize(gc)
    for key, value in zip(('Fitness', 'Survivability'), values):
        assert isfinite(value), f'{key} is {value} but must be finite. Check your characterize().'
        assert value >= 0.0, f'{key} is {value} but must be >= 0.0. Check your characterize().'
        assert value <= 1.0, f'{key} is {value} but must be <= 1.0. Check your characterize().'
    return values


class gene_pool(genetic_material_store):
    """Store of transient genetic codes & associated data for a population.

    The gene_pool is responsible for:
        1. Populating calcuble entry fields.
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

    The public member self.pool is the local cache of gGC's.
    """
    # TODO: Default genomic_library should be local host
    def __init__(self, genomic_library, configs=_DEFAULT_CONFIGS):
        """Connect to or create a gene pool.

        The gene pool data persists in a postgresql database. Multiple
        instances of the gene_pool() class can connect to the same database
        (use the same configuration).

        Args
        ----
        genomic_library (genomic_library): Source of genetic material.
        config(pypgtable config): The config is deep copied by pypgtable.
        """
        super().__init__(node_label=_NL, left_edge_label=_LEL, right_edge_label=_REL)

        # pool is a python2 type dictionary-like object of ref:gGC
        # All gGC's in pool must be valid & any modified are sync'd to the gene pool table
        # in the database at the end of the target epoch.
        self.pool = gene_pool_cache()
        _logger.info('Gene Pool Cache created.')

        self._gl = genomic_library
        self._pool = table(configs['gp'])
        _logger.info('Established connection to Gene Pool table.')

        # TODO: This should select from the local cache if it can then the DB table.
        self.select = self._pool.select

        # UID for this worker
        self.worker_id = uuid4()
        _logger.info(f'Worker ID: {self.worker_id}')

        # FIXME: Validators for all tables.

        # A dictionary of metric tables. Metric tables are undated at the end of the
        # taregt epoch.
        self._metrics = {c: table(configs[c]) for c in configs.keys() if 'metrics' in c}
        _logger.info('Established connections to metric tables.')

        # ?
        self._env_config = None
        self._population_data = {}
        self._populations_table = table(configs['populations'])
        self._owner_counters = {}
        self.reference = partial(reference, counters=self._owner_counters)
        # FIXME: Start population UID's at 0 with https://www.postgresql.org/docs/9.1/sql-altersequence.html
        if self._populations_table.raw.creator:
            _logger.info(f'This worker ({self.worker_id}) is the Populations table creator.')
        self.spuid = self.get_next_spuid()
        self.populations()
        _logger.info('Population(s) established.')

        # Modify the update strings to use the right table for the gene pool.
        self._update_str = UPDATE_STR.replace('__table__', configs['gp']['table'])

        # Used to track the number of updates to individuals in a pGC layer.
        self.layer_evolutions = [0] * NUM_PGC_LAYERS

        # If this instance created the gene pool then it is responsible for configuring
        # setting up database functions and initial population.
        if self._pool.raw.creator:
            _logger.info(f'This worker ({self.worker_id}) is the Gene Pool table creator.')
            self._pool.raw.arbitrary_sql(sql_functions(), read=False)

    def get_next_owner_id(self):
        """Get the next available owner UID."""
        return next(self._populations_table.update(_LAST_OWNER_ID_UPDATE_SQL, _LAST_OWNER_ID_QUERY_SQL,
            returning=('last_owner_id',), container='tuple'))[0]

    def purge_owner_counters(self):
        """When the owners no longer exist clean up the counters"""
        self._owner_counters = {}

    def _populate_local_cache(self) -> None:
        """Gather the latest and greatest from the GP.

        1. Load all the codons.
        2. For each population select the population size with the highest survivability.
        3. If not enough quality population pull from higher layer or create eGC's.
        4. Pull in the pGC's that created all the selected population & sub-gGCs
        5. Pull in the best pGC's at each layer.
        """
        self.pool.update({codon['ref']:codon for codon in self._pool(_LOAD_CODONS_SQL)})
        for population in filter(lambda x: 'characterize' in x, self._population_data.values()):
            literals = {'limit': population['size'], 'population_uid': population['uid']}
            for num, ggc in enumerate(self._pool.select(_LOAD_GPC_SQL, literals), start=1):
                self.pool[ggc['ref']] = ggc
            self._new_population(population, population['size'] - num)
            for ggc in self._pool.recursive_select(_REF_SQL, {'matches': list(self.pool.keys())}):
                self.pool[ggc['ref']] = ggc
            literals = {'limit': population['size']}
            for layer in range(NUM_PGC_LAYERS):
                literals['exclusions'] = self.pool.pGC_refs()
                literals['layer'] = layer
                for pgc in self._pool.select(_LOAD_PGC_SQL, literals):
                    self.pool[pgc['ref']] = pgc

    def populations(self) -> Dict[str, Dict[str, Any]]:
        """Return the definition of all populations.

        The gene pool object stores a local copy of the population data with
        fitness & diversity functions defined (in 0 to all) cases.
        """
        self._population_data.update({p['idx']: p for p in self._populations_table.select()})
        if _LOG_DEBUG:
            _logger.debug('Populations table:\n'+str("\n".join(self._population_data.values())))
        return {p['name']: p for p in self._population_data}

    def create_population(self, config:Dict = {}) -> Dict[str, Any]:
        """Create a population in the gene pool.

        The gene pool can support multiple populations.
        Once a population is created it cannot be modified.

        The population entry is initially created with a size of _POPULATION_IN_DEFINITION.
        This indicates the population is being created in the gene pool. Once complete the size
        is updated to the configured size.

        If the population entry already exists it will either be defined (gene pool populated)
        and the size entry will be set to something other than _POPULATION_IN_DEFINITION or
        it will be in definition in which case the method waits indefinately checking every
        1.0s (on average) until size changes from _POPULATION_IN_DEFINITION.

        Args
        ----
        config (dict): (required)
            'size':(int): Defines the population size.
            'name':(str): An unique short (<= 64 charaters) string giving a name.
            'inputs':iter(vtype): Target individual input definition using vt type.
            'outputs':iter(vtype): Target individual output definition using vt type.
            'characterize':callable(population): Characterize the population.
                The callable must take a single parameter that is an iterable of gGC's.
                The callable returns an iterable of characterisation dicts in the same order.
            'vt':(vtype): A vtype value defining how to interpret inputs & outputs.
            'description':(str): An arbitary string with a longer (<= 8192 characters) description.
            'meta_data':(str): Additional data about the population stored as a string (unlimited length).

        Returns
        -------
        data (dict):
            'idx':(int): A unique id given to the population.
            'worker_id:(int): The id of the worker that created the initial population.
            'size':(int): Defines the population size.
            'name':(str): An unique short (<= 64 charaters) string giving a name.
            'inputs':list(str): Target individual input definition as strings.
            'outputs':list(str): Target individual output definition as strings.
            'characterize':callable(gc): Characterize an individual of the population.
            'recharacterize': callable(g): Re-calculate survivability for a characterized individual.
            'vt':(vtype): vtype.EP_TYPE_STR.
            'description':(str): An arbitary string with a longer (<= 8192 characters) description.
            'meta_data':(str): Additional data about the population stored as a string (unlimited length).
            'created':(datetime): The time at which the population entry was created.
        """
        data = deepcopy(config)
        data['inputs'] = [asstr(i, config['vt']) for i in config['inputs']]
        data['outputs'] = [asstr(o, config['vt']) for o in config['outputs']]
        _, input_types, inputs = interface_definition(config['inputs'], config['vt'])
        _, output_types, outputs = interface_definition(config['outputs'], config['vt'])
        data['oih'] = ordered_interface_hash(input_types, output_types, inputs, outputs)
        data['size'] = _POPULATION_IN_DEFINITION
        data['worker_id'] = self.worker_id
        data = next(self._populations_table.insert((data,), '*'))
        data['vt'] = vtype.EP_TYPE_STR
        data['characterize'] = partial(_characterization_wrapper, characterize=config['characterize'])
        data['recharacterize'] = config['recharacterize']

        # If this worker created the entry then populate the gene pool.
        if data['worker_id'] == self.worker_id:
            _logger.info(f"Creating new population: {data['name']}.")
            self._population_data[data['uid']] = data
            self._populate(data['inputs'], data['outputs'], data['uid'], num=config['size'], vt=data['vt'])
            _logger.debug('Back to population create scope.')
            data['size'] = config['size']
            self._populations_table.update("{size} = {s}", "{uid} = {i}", {'s': data['size'], 'i': data['uid']})
        else:
            while data['size'] == _POPULATION_IN_DEFINITION:
                _logger.info(f"Waiting for population {data['name']} to be created by worker {data['worker_id']}.")
                sleep(0.5 + random())
                data = next(self._populations_table.select('{name} = {n}', {'n': data['name']}))
        return data

    def register_characterization(self, population_uid:int, characterize:Callable, recharacterize:Callable):
        """Register a function to characterize an individual & recharactize as the population changes.

        Every population in the Gene Pool that is to be evolved must have both a characterization
        function and a recharacterization function defined. The characterization function takes an individual
        gGC and calculates a fitness score and a survivability score. Both fitness
        & survivability are values between 0.0 and 1.0 inclusive. The recharacterization function
        calculates the survivability of a previously characterized gGG.

        Fitness is the fitness of the individual to the solution. A value of 1.0 means the
        solution is good enough & evolution ends.

        Survivability is often strongly corrolated with fitness but is not the same. Survivability
        is the relative weight of the indivdiual in the population for surviving to the next generation.
        It is different from fitness to allow diversity & novelty to be explored (which may lead
        unintuitively to greater fitness.)  The assumption is
        that survivability is a function of the population and not just the individual so as the
        population mutates the survivability of the individual changes.

        Args
        ----
        population_uid (int): The population UID to register the characterisation function to
        characterize (f(ggc)->tuple(float, float)
        recharacterize (f(ggc)->float)
        """
        self._population_data[population_uid]['characterize'] = characterize
        self._population_data[population_uid]['recharacterize'] = recharacterize

    def _new_population(self, population:Dict[str, Any], num:int):
        """Fetch or create num target GC's & recursively pull in the pGC's that made them.

        Construct a population with inputs and outputs as defined by inputs, outputs & vt.
        Inputs & outputs define the GC's interface.

        There are 2 sources of valid individuals: The higher layer (GL in this case) or to 
        create them.
        FIXME: Add pull from higher layer (recursively).
        FIXME: Implement forever work.

        Args
        ----
        population: Population definition.
        num: The number of GC's to create
        vt: See vtype definition.
        """
        # Check the population index exists
        _logger.info(f"Adding {num} GC's to population index: {population['uid']}.")

        # If there was not enough fill the whole population create some new gGC's & mark them as individuals too.
        # This may require pulling new agc's from the genomic library through steady state exceptions
        # in the stabilise() in which case we need to pull in all dependents not already in the
        # gene pool.
        _logger.info(f'{num} GGCs to create.')
        for _ in range(num):
            egc = eGC(inputs=population['inputs'], outputs=population['outputs'], vt=population['vt'])
            rgc, fgc_dict = stablize(self._gl, egc)

            # Just in case it is trickier than expected.
            retry_count = 0
            while rgc is None:
                _logger.info('eGC random creation failed. Retrying...')
                retry_count += 1
                egc = eGC(inputs=population['inputs'], outputs=population['outputs'], vt=population['vt'])
                rgc, fgc_dict = stablize(self._gl, egc)
                if retry_count == 3:
                    raise ValueError(f"Failed to create eGC with inputs = {population['inputs']} and outputs"
                                     f" = {population['outputs']} {retry_count} times in a row.")

            rgc['population_uid'] = population['uid']
            self.pool[rgc['ref']] = rgc
            self.pool.update(fgc_dict)
            _logger.debug(f'Created GGCs to add to Gene Pool: {[ref_str(ggc["ref"]) for ggc in (rgc, *fgc_dict.values())]}')

    def pull_from_gl(self, signatures, population_uid=None):
        """Pull aGCs and all sub-GC's recursively from the genomic library to the gene pool.

        aGC's are converted to gGC's.
        Higher layer fields are updated.
        Nodes & edges are added the the GP graph.
        SHA256 signatures are replaced by GP references.

        NOTE: This *MUST* be the only function pulling GC's into the GP from the GL.

        Args
        ----
        signatures (iterable(bytes[32])): Signatures to pull from the genomic library.
        population_uid: (int or None): Population UID to label ALL top level GC's with
        """
        if _LOG_DEBUG:
            _logger.debug(f'Recursively pulling {signatures} into Gene Pool for population {population_uid}.')
        gcs = tuple(self._gl.recursive_select(_SIGNATURE_SQL, {'matches': signatures}, _GL_COLUMNS))



        ggcs = gGC([gc for gc in gcs if gc['signature'] in signatures], population_uid=population_uid)
        self._gl.hl_copy(ggcs)
        if _LOG_DEBUG:
            _logger.debug("Adding sub-GC's")
        ggcs = gGC((gc for gc in gcs if gc['signature'] not in signatures))
        self._gl.hl_copy(ggcs)

    def push_to_gp(self):
        """Insert or update locally modified gGC's into the persistent gene_pool.

        NOTE: This *MUST* be the only function pushing GC's to the persistent GP.
        """
        # TODO: This can be optimised to further minimise the amount of data munging of unmodified values.
        # TODO: Check for dodgy values that are not just bad logic e.g. overflows
        modified_gcs = [gc for gc in filter(_MODIFIED_FUNC, self.pool.values())]
        if _LOG_DEBUG:
            _logger.debug(f'Validating GP DB entries.')
            for gc in modified_gcs:
                if not gp_entry_validator(dict(gc)):
                    _logger.debug(','.join([f'{k}: {type(v)}({v})' for k, v in gc.items()]))
                    _logger.error(f'gGC invalid:\n{gp_entry_validator.error_str()}.')
                    raise ValueError('gGC is invalid. See log.')

        # Add to the node graph
        self.add_nodes(modified_gcs)

        # FIXME: Use excluded columns depending on new or modified and pGC or not.
        for updated_gc in self._pool.upsert(modified_gcs, self._update_str, {}, _UPDATE_RETURNING_COLS):
            gc = self.pool[updated_gc['ref']]
            gc.update(updated_gc)
            for col in HIGHER_LAYER_COLS:  # FIXME: Wrong definition - should be GP higher layer cols & use hl_copy().
                gc[col] = gc[col[1:]]

        for gc in modified_gcs:
            gc['modified'] = False

    def delete_from_gp_cache(self, refs):
        """Delete GCs from the gene pool.

        If ref is not in the pool it is ignored.
        If ref is aliased by a signature both the signature & ref entries are deleted.

        Args
        ----
        refs(iterable(int)): GC 'ref' values to delete.
        """
        # TODO: Does this get rid of orphaned sub-GC's & otherwise unused pGC's?
        refs = self.remove_nodes(refs)
        _logger.info(f'Removing {len(refs)} GCs from GP local cache.')
        for ref in refs:
            # The gGC object cleans up all its memory (including the executable function)
            del self.pool[ref]
        # if refs:
            # FIXME: How does this work if another worker is using the GC?
            # 7-May-2022: I think the GP DB is only cleaned up in the parent process which
            # relieves issues for sub-processes. Parallel (remote) processes using the
            # same GP DB upsert so there is no risk.
            # self._pool.delete('{ref} in {ref_tuple}', {'ref_tuple': refs})

    def individuals(self, identifier=0):
        """Return a generator of individual gGCs for a population.

        The population is identified by identifier.

        Args
        ----
        identifier (str or int): Either the population name or index.

        Returns
        -------
        generator(gGC) of individuals in the population.
        """
        if isinstance(identifier, str):
            identifier = [p for p in self._population_data if p['name'] == identifier][0]['idx']
        return (gc for gc in filter(lambda x: x['population_uid'] == identifier, self.pool.values()))

    def cull_population(self, population_uid):
        """Reduce the target population to a locally managable size.

        TODO: 'Managable size' must be intelligently defined and overridable by the user.
        TODO: A way more efficient data structure is needed!

        The GC's culled are those with the lowest survivabilities.

        Args
        ----
        population_uid (int): The UID of the population to trim.
        """
        population = list((gc['ref'], gc['survivability'])
                          for gc in self.pool.values() if gc['population_uid'] == population_uid)
        if len(population) > _MAX_POPULATION_SIZE:
            num_to_cull = len(population) - _MAX_POPULATION_SIZE
            population.sort(key=lambda x: x[1])
            victims = tuple(ref for ref, _ in population[0:num_to_cull])
            if any((self.pool[ref]['modified'] for ref in victims)):
                _logger.debug('Modified population GCs to be purged from local cache. Pushing to GP.')
                self.push_to_gp()
            _logger.debug(f'{len(self.delete_from_gp_cache(victims))} GCs purged from population {population_uid}')

    def cull_physical(self):
        """Reduce the PGC population to a locally managable size.

        TODO: 'Managable size' must be intelligently defined and overridable by the user.
        TODO: A way more efficient data structure is needed!

        The PGC's culled are those with the lowest survivabilities.

        PGCs may be used in multiple layers and each layer has a limit. If a PGC
        is used in multiple layers that is a bonus.
        """
        victims = []
        safe = set()

        # FIXME: Very expensive
        pgcs = tuple(gc['ref'] for gc in self.pool.values() if is_pgc(gc))
        if len(pgcs) > _MAX_PGC_LAYER_SIZE * NUM_PGC_LAYERS:
            for layer in reversed(range(NUM_PGC_LAYERS)):
                layer_pgcs = [(ref, self.pool[ref]['pgc_survivability'][layer])
                              for ref in pgcs if self.pool[ref]['f_valid'][layer]]
                if len(layer_pgcs) > _MAX_PGC_LAYER_SIZE:
                    num_to_cull = len(layer_pgcs) - _MAX_PGC_LAYER_SIZE
                    layer_pgcs.sort(key=lambda x: x[1])
                    safe.update((ref for ref, _ in layer_pgcs[num_to_cull:]))
                    victims.extend((ref for ref, _ in layer_pgcs[0:num_to_cull] if ref not in safe))
            if any((self.pool[ref]['modified'] for ref in victims)):
                _logger.debug('Modified PGCs to be purged from local cache. Pushing to GP.')
                self.push_to_gp()
            _logger.debug(f'{len(self.delete_from_gp_cache(victims))} GCs purged from PGC population.')

    def _active_population_selection(self, population_uid):
        """Select a subset of the population to evolve.

        Args
        ----
        population_uid(int): The UID of the population to select from.

        Returns
        -------
        list(int): Refs of selected individuals.
        """
        refs = array(tuple(gc['ref'] for gc in self.pool.values() if gc['population_uid'] == population_uid))
        population_size = self._population_data[population_uid]['size']
        min_survivability = finfo(float32).tiny * population_size
        assert len(refs) >= population_size, (f'Population {population_uid} only has {len(refs)}'
                                              f' individuals which is less than the population size of {population_size}!')
        survivability = array(tuple(gc['survivability'] for gc in self.pool.values()
                                    if gc['population_uid'] == population_uid), dtype=float32)
        underflow_risk = survivability[logical_and(survivability < min_survivability, survivability != 0.0)]
        if any(underflow_risk):
            _logger.warning(f'{underflow_risk.sum()} survivability values risk underflow. \
                Risk free non-zero minimum is {min_survivability}')
            survivability[underflow_risk] = min_survivability
        num_survivors = count_nonzero(survivability)

        # There were no survivors.
        # Return a random selection.
        if not num_survivors:
            return choice(refs, population_size, False)

        # If there are less survivors than the population size return all survivors
        # and some randomly chosen dead
        if num_survivors < population_size:
            survivors = refs[survivability > 0.0]
            dead = choice(refs[survivability == 0.0], population_size - num_survivors, False)
            return concatenate((survivors, dead))
        
        # Otherwise pick the most likely to survive
        return refs[argsort(survivability)[:-population_size]]

    def viable_individual(self, individual, population_oih):
        """Check if the individual is viable as a member of the population.

        This function does static checking on the viability of the individual
        as a member of the population.

        Args
        ----
        individual (gGC): The gGC of the individual.
        population_oih (int): The ordered interface hash for the population the individual is to be a member of.

        Returns
        -------
        (bool): True if the individual is viable else False.
        """
        if _LOG_DEBUG:
            _logger.debug(f'Potentially viable individual {individual}')

        if individual is None:
            return False

        # Check the interface is correct
        individual_oih = ordered_interface_hash(individual['input_types'], individual['output_types'],
                                                individual['inputs'], individual['outputs'])

        if _LOG_DEBUG:
            _logger.debug(f"Individual is {('NOT ', '')[population_oih == individual_oih]}viable.")
        return population_oih == individual_oih

    def generation(self, population_uid):
        """Evolve the population one generation and characterise it.

        Evolutionary steps are:
            a. Select a population size group of individuals weighted by survivability. Selection
               is from every individual in the local GP cache that is part of the population.
            b. For each individual:
                1. Select a pGC to operate on the individual
                2. Evolve the individual to produce an offspring
                    TODO: Inheritance
                3. Characterise the offspring
                4. Update the individuals (parents) parameters (evolvability, survivability etc.)
                5. Update the parameters of the pGC (recursively)
            c. Reassess survivability for the entire population in the local cache.
                    TODO: Optimisation mechanisms

        Args
        ----
        population_uid (int): The index of the target population to evolve
        """
        start = time()
        characterize = self._population_data[population_uid]['characterize']
        active = self._active_population_selection(population_uid)
        population_oih = self._population_data[population_uid]['oih']

        if _LOG_DEBUG:
            _logger.debug(f'Evolving population {population_uid}')

        pgcs = select_pGC(self, active)
        for count, (individual_ref, pgc) in enumerate(zip(active, pgcs)):
            individual = self.pool[individual_ref]
            if _LOG_DEBUG:
                _logger.debug(f'Individual ({count + 1}/{len(pgcs)}): {individual}')
                _logger.debug(f"Mutating with pGC {pgc['ref']}")

            wrapped_pgc_exec = create_callable(pgc, self.pool)
            result = wrapped_pgc_exec((individual,))
            if result is None:
                # pGC went pop - should not happen very often
                _logger.warning(f"pGC {ref_str(pgc['ref'])} threw an exception when called.")
                offspring = None
            else:
                offspring = result[0]

            if _LOG_DEBUG:
                _logger.debug(f'Offspring ({count + 1}/{len(pgcs)}): {offspring}')

            if offspring is not None and self.viable_individual(offspring, population_oih):
                offspring['exec'] = create_callable(offspring, self.pool)
                new_fitness, survivability = characterize(offspring)
                del offspring['exec']
                offspring['fitness'] = new_fitness
                offspring['survivability'] = survivability
                population_GC_inherit(offspring, individual, pgc)
                delta_fitness = new_fitness - individual['fitness']
                population_GC_evolvability(individual, delta_fitness)
            else:
                # PGC did not produce an offspring.
                delta_fitness = -1.0
            pgc_evolutions = pGC_fitness(self, pgc, delta_fitness)

        # Update survivabilities as the population has changed
        if _LOG_DEBUG:
            _logger.debug('Re-characterizing population.')
        population = tuple(gc for gc in self.pool.values() if gc['population_uid'] == population_uid)
        self._population_data[population_uid]['recharacterize'](population)

        # Pushing could be expensive. Larger batches are more efficient but could cause a
        # bigger data loss. May be let the user decide. Regardless we must push any culled GCs.
        self.push_to_gp()

        # This is just about memory management. Any culled GC's are automatically pushed
        # to the persistent GP if they are not there already.
        self.cull_population(population_uid)
        self.cull_physical()
        self.metrics(population_uid, pgcs, time()-start)

    def metrics(self, population_uid, pgcs, duration):
        """Calculate and record metrics.

        Called one per generation as the last function.

        Args
        ----
        duration (float): Number of seconds it took the last generation to execute.
        pgcs (iter(gc)): Valid pGC's in the Gene Pool Cache.
        population_uid (int): The index of the target population to evolve
        """
        self.population_metrics(population_uid, duration)
        self.pgc_metrics(pgcs, duration)
        self.gp_metrics(duration)

    def population_metrics(self, population_uid, duration):
        """Target metrics."""
        # TODO: Define constants for field names
        fitness = [individual['fitness'] for individual in self.individuals(population_uid)]
        evolvability = [individual['evolvability'] for individual in self.individuals(population_uid)]
        survivability = [individual['survivability'] for individual in self.individuals(population_uid)]
        generation = [individual['generation'] for individual in self.individuals(population_uid)]
        gc_count = [individual['num_codes'] for individual in self.individuals(population_uid)]
        c_count = [individual['num_codons'] for individual in self.individuals(population_uid)]
        self._metrics['population_metrics'].insert([{
            'population_uid': population_uid,
            'count': len(fitness),
            'f_max': max(fitness),
            'f_mean': mean(fitness),
            'f_min': min(fitness),
            'e_max': max(evolvability),
            'e_mean': mean(evolvability),
            'e_min': min(evolvability),
            's_max': max(survivability),
            's_mean': mean(survivability),
            's_min': min(survivability),
            'g_max': max(generation),
            'g_mean': mean(generation),
            'g_min': min(generation),
            'gcc_max': max(gc_count),
            'gcc_mean': mean(gc_count),
            'gcc_min': min(gc_count),
            'cc_max': max(c_count),
            'cc_mean': mean(c_count),
            'cc_min': min(c_count),
            'eps': self._population_data[population_uid]['size'] / duration,
            'tag': 0,
            'worker_id': 0}])

    def pgc_metrics(self, pgcs, duration):
        """Per pGC layer metrics."""
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
                'eps': evolutions / duration,
                'performance': 0.0,
                'tag': 0,
                'worker_id': 0}])

    def gp_metrics(self, duration):
        """Per pGC layer metrics."""
        pass
