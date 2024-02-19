"""Gene pool management for Erasmus GP."""
from __future__ import annotations
from copy import deepcopy
from json import load
from logging import DEBUG, Logger, NullHandler, getLogger
from os.path import dirname, join
from typing import Any, TYPE_CHECKING, cast, Iterable
from functools import partial

from egp_types.conversions import compress_json, decompress_json, memoryview_to_bytes
from egp_types.gc_type_tools import NUM_PGC_LAYERS
from egp_types.xgc_validator import gGC_entry_validator
from egp_types.genetic_code import genetic_code
from egp_utils.common import merge, default_erasumus_db_config
from pypgtable import table
from pypgtable.pypgtable_typing import Conversions, TableConfigNorm, TableSchema
from pypgtable.validators import raw_table_config_validator

from .gene_pool_cache import gene_pool_cache, GPC_HIGHER_LAYER_COLS

from .gene_pool_common import (
    GP_HIGHER_LAYER_COLS,
    GP_RAW_TABLE_SCHEMA,
    GP_UPDATE_RETURNING_COLS,
)
from .genetic_material_store import genetic_material_store, _PTR_MAP_PLUS_PGC, UPDATE_STR, _PTR_MAP
from .genomic_library import (
    GL_HIGHER_LAYER_COLS,
    GL_RAW_TABLE_SCHEMA,
    genomic_library,
    gl_sql_functions,
)
from .egp_typing import GenePoolConfig, GenePoolConfigNorm


# For layer violation type definitions.
if TYPE_CHECKING:
    from egp_population.egp_typing import PopulationConfigNorm


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
# TODO: Add a _LOG_CONSISTENCY which additionally does consistency checking
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


_GP_CONVERSIONS: Conversions = (
    ("graph", compress_json, decompress_json),
    ("meta_data", compress_json, decompress_json),  # TODO: Why store this?
    ("inputs", None, memoryview_to_bytes),
    ("outputs", None, memoryview_to_bytes)
)


with open(
    join(dirname(__file__), "formats/gp_metrics_table_format.json"),
    "r",
    encoding="utf8",
) as file_ptr:
    _GP_METRICS_TABLE_SCHEMA: TableSchema = load(file_ptr)
with open(
    join(dirname(__file__), "formats/gp_pgc_metrics_table_format.json"),
    "r",
    encoding="utf8",
) as file_ptr:
    _GP_PGC_METRICS_TABLE_SCHEMA: TableSchema = load(file_ptr)
with open(join(dirname(__file__), "formats/gp_meta_table_format.json"), "r", encoding="utf8") as file_ptr:
    _GP_META_TABLE_SCHEMA: TableSchema = load(file_ptr)
with open(
    join(dirname(__file__), "formats/gp_pgc_probabilities_table_format.json"),
    "r",
    encoding="utf8",
) as file_ptr:
    _GP_PGC_PROB_TABLE_SCHEMA: TableSchema = load(file_ptr)


# GC queries
_LOAD_GL_COLUMNS: tuple[str, ...] = tuple(k for k in GL_RAW_TABLE_SCHEMA.keys() if k not in GL_HIGHER_LAYER_COLS)
_LOAD_GP_COLUMNS: tuple[str, ...] = tuple(k for k in GP_RAW_TABLE_SCHEMA.keys() if k not in GP_HIGHER_LAYER_COLS)
_SIGNATURE_SQL: str = "WHERE ({signature} = ANY({matches}))"
_LOAD_GPC_SQL: str = "WHERE {population_uid} = {puid} ORDER BY {survivability} DESC LIMIT {limit}"
_LOAD_PGC_SQL: str = (
    "WHERE {pgc_f_count}[{layer}] > 0 AND NOT({signature} = ANY({exclusions})) " "ORDER BY {pgc_fitness}[{layer}] DESC LIMIT {limit}"
)
_LOAD_CODONS_SQL: str = "WHERE {creator}::text = '22c23596-df90-4b87-88a4-9409a0ea764f'"
_META_DEFAULTS_SQL = "INSERT INTO {mdt} DEFAULT VALUES;"


# The gene pool default config
_DEFAULT_GP_CONFIG: TableConfigNorm = raw_table_config_validator.normalized(
    {
        "database": default_erasumus_db_config(),
        "table": "gene_pool",
        "ptr_map": _PTR_MAP,
        "schema": GP_RAW_TABLE_SCHEMA,
        "create_table": True,
        "create_db": True,
        "conversions": _GP_CONVERSIONS,
    }
)
_DEFAULT_GP_METRICS_CONFIG: TableConfigNorm = raw_table_config_validator.normalized(
    {
        "database": default_erasumus_db_config(),
        "table": "gp_metrics",
        "schema": _GP_METRICS_TABLE_SCHEMA,
        "create_table": False,
        "create_db": False,
    }
)
_DEFAULT_PGC_METRICS_CONFIG: TableConfigNorm = raw_table_config_validator.normalized(
    {
        "database": default_erasumus_db_config(),
        "table": "pgc_metrics",
        "schema": _GP_PGC_METRICS_TABLE_SCHEMA,
        "create_table": False,
        "create_db": False,
    }
)
_DEFAULT_GP_META_CONFIG: TableConfigNorm = raw_table_config_validator.normalized(
    {
        "database": default_erasumus_db_config(),
        "table": "meta_data",
        "schema": _GP_META_TABLE_SCHEMA,
        "create_table": False,
        "create_db": False,
    }
)
_DEFAULT_CONFIGS: GenePoolConfigNorm = {
    "gene_pool": _DEFAULT_GP_CONFIG,
    "meta_data": _DEFAULT_GP_META_CONFIG,
    "gp_metrics": _DEFAULT_GP_METRICS_CONFIG,
    "pgc_metrics": _DEFAULT_PGC_METRICS_CONFIG,
}


def default_config() -> GenePoolConfigNorm:
    """Return a deepcopy of the default genomic library configuration.

    The copy may be modified and used to create a genomic library instance.

    Returns
    -------
    The default genomic_library() configuration.
    """
    return deepcopy(_DEFAULT_CONFIGS)


def gp_sql_functions() -> str:
    """Load the SQL functions used by the gene pool & dependent repositiories of genetic material."""
    with open(join(dirname(__file__), "data/gp_functions.sql"), "r", encoding="utf-8") as fileptr:
        return fileptr.read()


class gene_pool(genetic_material_store):
    """Store of transient genetic codes & associated data for a population.

    The gene_pool is responsible for:
        1. Populating calculable entry fields.
        2. Providing an application interface to the fields.
        3. Persistence of updates to the gene pool.
        4. Local caching of GC's.

    The gene pool must be consistent i.e. no entry can depend on a genetic code
    that is not in the gene_pool.

    The primary difference with the genomic_library is the presence of transient data.
    For the same reasons validation is not performed unless in debug mode.

    The gene_pool is more local (faster to access) than the genomic library
    due to the heavy transaction load. It is also designed to be multi-process
    memory efficient.

    The public member self.pool is the local cache of gGC's. Querying or writing to
    the GP should only happen during an egp_physics.physics.proximity_search() or
    when filling / flushing the GPC.
    """

    # TODO: Default genomic_library should be local host

    def __init__(
        self,
        populations: dict[int, PopulationConfigNorm],
        glib: genomic_library,
        config: GenePoolConfig | GenePoolConfigNorm,
    ) -> None:
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
        # Normalize the table configs
        def normalize(k, v) -> dict[str, Any]:
            return merge(v, raw_table_config_validator.normalized(config.get(k, {})), no_new_keys=True, update=True)
        self.config: GenePoolConfigNorm = cast(GenePoolConfigNorm,{k: normalize(k, v) for k, v in _DEFAULT_CONFIGS.items()})
        super().__init__(self.config["gene_pool"], [])

        self._populations: dict[int, PopulationConfigNorm] = deepcopy(populations)
        self.glib: genomic_library = glib
        self.pool: gene_pool_cache = gene_pool_cache(2**20, partial(_update, gp=self))
        genetic_code.set_gpc(self.pool)

        self._tables: dict[str, table] = {"meta_data": table(self.config["meta_data"])}
        if self._tables["meta_data"].raw.creator:
            literals: dict[str, Any] = {"mdt": self._tables["meta_data"].raw.config["table"]}
            self._tables["meta_data"].raw.arbitrary_sql(_META_DEFAULTS_SQL.format(**literals), literals=literals, read=False)

        # If this process did not create the gene pool table the following line will wait
        # for the other tables to be created.
        self._tables.update({k: table(v) for k, v in self.config.items()})
        _logger.info("Established connections to gene pool tables.")
        self.select = self._tables["gene_pool"].select

        # Modify the update strings to use the right table for the gene pool.
        self.update_str: str = UPDATE_STR.replace("__table__", self.config["gene_pool"]["table"])

        # Fill the local cache with populations in the persistent GP.
        self._populate_local_cache()

    def _creator(self, _: list[str]) -> None:
        # If this instance created the gene pool then it is responsible for configuring
        # setting up database functions and initial population.
        self.library.raw.arbitrary_sql(gl_sql_functions(), read=False)
        self.library.raw.arbitrary_sql(gp_sql_functions(), read=False)
        # Enable creation of the other GP tables
        for table_config in cast(Iterable[TableConfigNorm], self.config.values()):
            if "create_table" in table_config:
                table_config["create_table"] = True

    def sub_process_init(self) -> None:
        """Define subprocess specific values *WHEN IN THE SUB-PROCESS*."""
        self.library.raw.creator = False

    def creator(self) -> bool:
        """True if this process created the gene pool table in the database."""
        return self.library.raw.creator

    def _populate_local_cache(self) -> None:
        """Gather the latest and greatest from the GP.
        1. Load all the codons from the genomic library (this ensures we get the all).
        2. For each population select the population size with the highest survivability.
        3. Pull in the pGC's that created all the selected population & sub-gGCs
        4. Pull in the best pGC's at each layer.
        5. If not enough quality population pull from higher layer or create eGC's.
        """
        _logger.info("Populating local cache from Gene Pool.")
        self.pull(list(self.glib.select(_LOAD_CODONS_SQL, columns=("signature",), container="tuple")))
        for population in self._populations.values():
            literals: dict[str, Any] = {"limit": population["size"], "puid": population["uid"]}
            self.pool.update(self.library.select(_LOAD_GPC_SQL, literals, _LOAD_GP_COLUMNS))
            self.library.raw.ptr_map_def(_PTR_MAP_PLUS_PGC)
            self.pool.update(self.library.recursive_select(_SIGNATURE_SQL, {"matches": list(self.pool.keys())}))
            self.library.raw.ptr_map_def(_PTR_MAP)
            literals = {"limit": population["size"]}
            for layer in range(NUM_PGC_LAYERS):
                literals["exclusions"] = list({v["pgc"]["signature"] for v in self.pool.values()})
                literals["layer"] = layer
                self.pool.update(self.library.select(_LOAD_PGC_SQL, literals))
        # Saves space by removing duplicate constant objects
        self.pool.optimize()

    def pull(self, signatures: list[memoryview], population_uid: int | None = None) -> None:
        """Pull aGCs and all sub-GC's recursively from the genomic library to the gene pool.
        LGC's are converted to gGC's and higher layer fields are updated.
        NOTE: This *MUST* be the only function pulling GC's into the GP from the GL.

        Args
        ----
        signatures: Signatures to pull from the genomic library.
        population_uid: Population UID to label ALL top level GC's with
        """
        if _LOG_DEBUG:
            _logger.debug(f"Recursively pulling {signatures} into Gene Pool for population {population_uid}.")
        batch: list[dict[str, Any]] = []
        for ggc in self.glib.recursive_select(_SIGNATURE_SQL, {"matches": signatures}, _LOAD_GL_COLUMNS):
            if population_uid is not None and ggc["signature"] in signatures:
                ggc["population_uid"] = population_uid

            # Set the higher layer fields
            # A higher layer field starts with an underscore '_' and has an underscoreless counterpart.
            # e.g. '_field' and 'field'. The _field holds the value of field when the GC was pulled from
            # the higher layer. i.e. after being pulled from the GMS _field must = field. field can then
            # be modified by the lower layer. NB: Updating the value back into the GMS is a bit more complex.
            ggc.update({k: ggc[k[1:]] for k in GP_HIGHER_LAYER_COLS})
            batch.append(ggc)

            # Batch push to GP as it is more efficient
            if len(batch) == 1024:
                self._pull(batch)
        # Push the remaining GC's
        self._pull(batch)

    def _pull(self, batch: list[dict[str, Any]]) -> None:
        """Helper method for the pull method. This is the batch insertion into the GP."""
        # The GP has its own higher layer columns for the GPC which are returned by the upsert.
        for ggc, update in zip(batch, self.library.upsert(batch, self.update_str, {}, GP_UPDATE_RETURNING_COLS)):
            ggc.update(update)
            ggc.update({k: ggc[k[1:]] for k in GPC_HIGHER_LAYER_COLS})
            self.pool.add(ggc)
            if _LOG_DEBUG:
                for k, v in ggc.items():
                    if isinstance(v, memoryview):
                        ggc[k] = bytes(v)
                if not gGC_entry_validator.validate(ggc):
                    _logger.error(ggc)
                    _logger.error(gGC_entry_validator.error_str())
                    assert False, "GGC is not valid!"
        # Empty the batch list
        batch.clear()


def _update(gp: gene_pool, ggcs: tuple[genetic_code, ...]) -> None:
    """Push the genetic codes into the persistent store of the gene pool."""

    # To keep pylance happy about 'possibly unbound'
    if _LOG_DEBUG:
        _logger.debug("Validating GPC GC's --> GP DB entries.")
        for _ in filter(lambda x: not gGC_entry_validator.validate(x), ggcs):
            _logger.error(f"gGC from GPC to be pushed to persistent GP is invalid:\n{gGC_entry_validator.error_str()}.")
            assert False, "gGC from GPC to be pushed to persistent GP is invalid"
    gp.library.upsert(ggcs, gp.update_str)
    # TODO: Add a check to see if the upsert updated the GP as expected. Can do this
    # by pulling the GC's back out and checking the values.