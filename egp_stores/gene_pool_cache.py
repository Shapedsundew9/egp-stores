"""Gene Pool Cache.

The gene pool cache is a space and time optimised store of GC's. It is designed to
be multi-process friendly.

Naively, the gene pool cache could be implemented as a dictionary with reference keys.
This would be fast but does not scale well. Python dictionaries use huge amounts
of memory and are updated in a spatially broad manner requiring subprocesses to maintain
an almost full copy even if most entries are only read.

The gene pool cache as implemented here maintains a dictionary like interface but takes
advantage of some GC structural design choices to efficiently store data in numpy arrays where possible.

NOTE: GPC does not preserve entry order like a python3 dictionary would.

It also takes advantage of GC usage patterns to cluster stable and volatile GC data which
makes efficient use of OS CoW behaviour in a multi-process environment as well as minimising
the volume of null data between GC variants by partitioning between gGC's & pGC's.

Rough benchmarking:

Assuming a GGC can be approximated by 125 integers in a dictionary and
100,000 GGCs in the Gene Pool Cache implemented as a dictionary:

gpc = {k:{v:v for v in tuple(range(125))} for k in tuple(range(100000))}

The memory used by python3 3.10.6 is 467 MB (4565 MB for 1,000,000)

Assuming a GGC can be represented by a dictionary of indexes into to a
numpy array of int64 and shape (125, 100000) then the memory used is

gpc_index = {k:k for k in tuple(range(100000))}
gpc = zeros((125, 1000000), dtype=int64)

The memory used by python3 3.10.6 is 10 + 100 = 110 MB. (1085 MB for 1,000,000)

That is a saving of 4x.

The saving get compunded when considering a dict of dict.
Actual results from a random 127 element GPC:
14:01:30 INFO test_gene_pool_cache.py 93 Dict size: sys.getsizeof = 4688 bytes, pympler.asizeof = 5399488 bytes.
14:01:30 INFO test_gene_pool_cache.py 94 GPC size: sys.getsizeof = 56 bytes, pympler.asizeof = 204576 bytes.

That is a saving of 25x.

For read-only GC's in the persistent Gene Pool loaded on startup ~75% of the data
is read only avoiding 4x as many CoW's giving a total factor of ~16x for that data.
Bit of an anti-pattern for python but in this case the savings are worth it.
"""

from copy import deepcopy
from functools import lru_cache
from json import load
from logging import DEBUG, Logger, NullHandler, getLogger
from os.path import dirname, join
from random import choice
from re import Match, search
from typing import Any, Callable, Generator, Literal, NoReturn

from egp_types.aGC import aGC
from egp_types.gc_type_tools import is_pgc
from egp_types.xGC import pGC, xGC
from egp_utils.base_validator import base_validator
from egp_utils.common import merge
from egp_utils.packed_store import Field, indexed_store, packed_store
from numpy import bool_, double, float32, int16, int32, int64
from numpy.random import choice as np_choice
from pypgtable.pypgtable_typing import SchemaColumn
from pypgtable.validators import PYPGTABLE_COLUMN_CONFIG_SCHEMA

from .gene_pool_cache_graph import gene_pool_cache_graph
from .gene_pool_common import GP_RAW_TABLE_SCHEMA

# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)

_REGEX: Literal["([A-Z]*)\\[{0,1}(\\d{0,})\\]{0,1}"] = r"([A-Z]*)\[{0,1}(\d{0,})\]{0,1}"


# TODO: Make a new cerberus validator for this extending the pypgtable raw_table_column_config_validator
# TODO: Definition class below should inherit from pypgtable (when it has a typeddict defined)


class ConfigDefinition(SchemaColumn):
    """GPC field configuration."""

    ggc_only: bool
    pgc_only: bool
    indexed: bool
    signature: bool
    init_only: bool
    reference: bool


# Load the GPC config definition which is a superset of the pypgtable column definition
GPC_FIELD_SCHEMA: dict[str, dict[str, Any]] = deepcopy(PYPGTABLE_COLUMN_CONFIG_SCHEMA)
with open(
    join(dirname(__file__), "formats/gpc_field_definition_format.json"),
    "r",
    encoding="utf8",
) as file_ptr:
    merge(GPC_FIELD_SCHEMA, load(file_ptr))
gpc_field_validator: base_validator = base_validator(
    GPC_FIELD_SCHEMA, purge_uknown=True
)
GPC_RAW_TABLE_SCHEMA: dict[str, dict[str, Any]] = deepcopy(GP_RAW_TABLE_SCHEMA)
with open(
    join(dirname(__file__), "formats/gpc_table_format.json"), "r", encoding="utf8"
) as file_ptr:
    merge(GPC_RAW_TABLE_SCHEMA, load(file_ptr))
GPC_TABLE_SCHEMA: dict[str, ConfigDefinition] = {
    k: gpc_field_validator.normalized(v) for k, v in GPC_RAW_TABLE_SCHEMA.items()
}
GPC_HIGHER_LAYER_COLS: tuple[str, ...] = tuple(
    (key for key in filter(lambda x: x[0] == "_", GPC_TABLE_SCHEMA))
)
GPC_UPDATE_RETURNING_COLS: tuple[str, ...] = tuple(
    (x[1:] for x in GPC_HIGHER_LAYER_COLS)
) + ("updated", "created")
GPC_REFERENCE_COLUMNS: tuple[str, ...] = tuple(
    (
        key
        for key, _ in filter(
            lambda x: x[1].get("reference", False), GPC_TABLE_SCHEMA.items()
        )
    )
)
_GGC_INIT_LAMBDA: Callable[..., bool] = lambda x: not x[1].get(
    "init_only", False
) and not x[1].get("pgc_only", False)
_PGC_INIT_LAMBDA: Callable[..., bool] = lambda x: not x[1].get(
    "init_only", False
) and not x[1].get("ggc_only", False)

# Lazy add igraph
sql_np_mapping: dict[str, Any] = {
    "BIGINT": int64,
    "BIGSERIAL": int64,
    "BOOLEAN": bool_,
    "DOUBLE PRECISION": double,
    "FLOAT8": double,
    "FLOAT4": float32,
    "INT": int32,
    "INT8": int64,
    "INT4": int32,
    "INT2": int16,
    "INTEGER": int32,
    "REAL": float32,
    "SERIAL": int32,
    "SERIAL8": int64,
    "SERIAL4": int32,
    "SERIAL2": int16,
    "SMALLINT": int16,
    "SMALLSERIAL": int16,
}


def create_cache_config(
    table_format_json: dict[str, ConfigDefinition]
) -> dict[str, Field]:
    """Converts the Gene Pool table format to a GPC config.

    | Field | Type | Default | Description |
    ------------------------------
    | volatile | bool | False | True if the field may be written to after creation |
    | ggc_only | bool | False | True if field only present in non-pGC GC's |
    | pgc_only | bool | False | True if field only present in pGC GC's |
    | indexed  | bool | False | Set to True for large sparsely defined fields |

    Sparsely defined fields with > 4 bytes storage (e.g. > int32) can save memory by
    being stored as a 32 bit index into a smaller store.

    Args
    ----
    table_format_json: The normalized GP pypgtable format JSON with additikonal fields defined above.

    Returns
    -------
    See _gpc class fields definition.
    """
    # TODO: Add classes to lookup common large collective fields like input*, output*
    # Thinking something like a hash stored per entry that can be used to look up the
    # specifics. Monitor the unique hash to entry ratio to make sure it is worth it.
    fields: dict[str, Field] = {}
    for column, definition in table_format_json.items():
        match: Match[str] | None = search(_REGEX, definition["type"])
        assert (
            match is not None
        ), f"definition['type'] = {definition['type']} which cannot be parsed!"
        typ: str = "LIST"
        if match is not None and "[]" not in definition["type"]:
            typ = match.group(1)
        length: int = 1 if not match.group(2) else int(match.group(2))
        default: str | int = (
            0
            if definition.get("default", "null") == "null"
            else definition.get("default", 0)
        )
        fields[column] = {
            "type": sql_np_mapping.get(typ, list)
            if not definition.get("indexed", False)
            else indexed_store,
            "length": length,
            "default": sql_np_mapping[typ](default) if typ in sql_np_mapping else None,
            "read_only": not definition.get("volatile", False),
            "read_count": 0,
            "write_count": 0,
        }
    return fields


class gene_pool_cache(gene_pool_cache_graph):
    """The Gene Pool Cache (GPC)."""

    # TODO: Implement bulk get, set & del methods

    def __init__(self, delta_size: int = 17) -> None:
        super().__init__()
        fields: dict[str, ConfigDefinition] = {
            k: v for k, v in filter(_GGC_INIT_LAMBDA, GPC_TABLE_SCHEMA.items())
        }
        self._ggc_cache: packed_store = packed_store(
            create_cache_config(fields), xGC, delta_size
        )
        self._ggc_refs: dict[int, int] = self._ggc_cache.ref_to_idx
        fields: dict[str, ConfigDefinition] = {
            k: v for k, v in filter(_PGC_INIT_LAMBDA, GPC_TABLE_SCHEMA.items())
        }
        self._pgc_cache: packed_store = packed_store(
            create_cache_config(fields), xGC, delta_size
        )
        self._pgc_refs: dict[int, int] = self._pgc_cache.ref_to_idx

    def __contains__(self, ref: int) -> bool:
        """Test if a ref is in the GPC.

        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        True if ref is present.
        """
        return ref in self._ggc_refs or ref in self._pgc_refs

    def __delitem__(self, ref: int) -> None:
        """Remove the entry for ref from the GPC.

        If any sub-GC's are orphaned they will also be removed.

        Args
        ----
        ref: GPC unique GC reference.
        """
        refs: list[int] = self.remove([ref])
        for _ref in refs:
            if _ref in self._ggc_refs:
                del self._ggc_cache[_ref]
            else:
                del self._pgc_cache[_ref]

    # TODO: Add cache stats to the log output.
    @lru_cache(maxsize=1024)
    def __getitem__(self, ref: int) -> xGC:
        """Return an xGC dict-like structure from the GPC.

        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        dict-like GC record.
        """
        if ref in self._ggc_refs:
            return self._ggc_cache[ref]
        return self._pgc_cache[ref]

    def get(self, ref: int, default: xGC | None = None) -> xGC | None:
        """Return an xGC dict-like structure from the GPC or default.

        Args
        ----
        ref: GPC unique GC reference.
        default: Return if ref not in the GPC.

        Returns
        -------
        xGC with ref or default
        """
        if ref in self._ggc_refs:
            return self._ggc_cache[ref]
        if ref in self._pgc_cache:
            return self._pgc_cache[ref]
        return default

    def __len__(self) -> int:
        """The number of entries."""
        return len(self._ggc_refs) + len(self._pgc_refs)

    def __setitem__(self, ref: int, value: aGC) -> None:
        """Create a gGC entry in the GPC.

        NOTE: Set of an existing entry behaves like an update()

        Args
        ----
        ref: The GC unique reference in the GP.
        value: A dict-like object defining at least the required fields of a gGC.
        """
        if is_pgc(value):
            self._pgc_cache[ref] = value  # type: ignore aGC is always dict compatible.
        else:
            self._ggc_cache[ref] = value  # type: ignore aGC is always dict compatible.
        self.add([value])

    def __copy__(self) -> NoReturn:
        """Make sure we do not copy the GPC."""
        assert False, "Shallow copy of GPC."

    def __deepcopy__(self, obj: Any) -> NoReturn:
        """Make sure we do not copy GPC."""
        assert False, "Deep copy of the GPC."

    def get_list(
        self, refs: list[int] | tuple[int, ...] | Generator[int, None, None]
    ) -> list[xGC]:
        """Return a list of xGCs from the GPC.

        Args
        ----
        refs: A list of GC references.

        Returns
        -------
        A list of xGCs.
        """
        return [
            self._ggc_cache[ref] if ref in self._ggc_refs else self._pgc_cache[ref]
            for ref in refs
        ]

    def find(
        self, field: str, value: Any, ggc_only: bool = True
    ) -> Generator[xGC, None, None]:
        """Find all GC's with a field matching value.

        Args
        ----
        field: The field name to match.
        value: The value to match.
        ggc_only: If True only gGC's are searched else only pGC's are searched as well. The most common
            use case is to only search gGC's of a problem population.

        Returns
        -------
        A generator of xGC's.
        """
        for ggc in self._ggc_cache.find(field, value):
            yield ggc
        if not ggc_only:
            for pgc in self._pgc_cache.find(field, value):
                yield pgc

    def keys(self) -> Generator[int, None, None]:
        """A view of the references in the GPC."""
        for key in self._ggc_refs:
            yield key
        for key in self._pgc_refs:
            yield key

    def is_pgc(self, ref) -> bool:
        """True if the ref is that of a pGC."""
        return ref in self._pgc_refs

    def pgc_refs(self) -> list[int]:
        """All pGC references in the GPC."""
        return list(self._pgc_refs.keys())

    def random_pgc(self, depth: int = 0) -> pGC:
        """A weighted random pGC from the GPC."""
        ref_allocs, fitness_allocs = self._pgc_cache.get_allocation(
            ("ref", "pgc_fitness")
        )
        allocation_sums = [fitness[depth].sum() for fitness in fitness_allocs]

        # No pGC's with non-zero fitness (i.e. all pGC's a unused codons)
        total_weight = sum(allocation_sums)
        if total_weight == 0:
            return self._pgc_cache[choice(tuple(self._pgc_refs.keys()))]

        # Weighted random selection of allocation
        allocation_idx = np_choice(
            tuple(range(len(allocation_sums))),
            p=[x / total_weight for x in allocation_sums],
        )
        refs = ref_allocs[allocation_idx]
        f: double = fitness_allocs[allocation_idx][depth]
        return self._pgc_cache[refs[np_choice(tuple(range(len(refs))), p=f / f.sum())]]

    def items(self) -> Generator[tuple[int, xGC], None, None]:
        """A view of the gGCs in the GPC."""
        for ref in self._ggc_refs:
            yield ref, self[ref]
        for ref in self._pgc_refs:
            yield ref, self[ref]

    def update(self, value) -> None:
        """Update the GPC with a dict-like collection of gGCs."""
        for k, v in value.items():
            self[k] = v

    def values(self) -> Generator[xGC, None, None]:
        """A view of the gGCs in the GPC."""
        for ref in self._ggc_refs:
            yield self[ref]
        for ref in self._pgc_refs:
            yield self[ref]

    def modified(self, all_fields: bool = False) -> Generator[xGC, None, None]:
        """A view of the modified gGCs in the GPC.

        Args
        ---
        all_fields: If True all the xGC fields are returned else only the writable fields.

        Returns
        -------
        Each modified xGC.
        """
        for ggc in self._ggc_cache.modified(all_fields):
            yield ggc
        for pgc in self._pgc_cache.modified(all_fields):
            yield pgc
