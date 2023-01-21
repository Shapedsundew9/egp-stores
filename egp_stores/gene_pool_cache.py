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
from __future__ import annotations

from collections.abc import KeysView as dict_keys
from copy import deepcopy
from functools import partial
from json import load
from logging import DEBUG, Logger, NullHandler, getLogger
from os.path import dirname, join
from re import Match, search
from typing import Any, Callable, Generator, Literal, NoReturn, Self

from egp_types.eGC import eGC
from egp_types.gc_type_tools import is_pgc
from egp_types.mGC import mGC
from egp_types.xGC import xGC, gGC, Field
from egp_types.reference import ref_str
from egp_utils.base_validator import base_validator
from egp_utils.common import merge
from numpy import bool_, float32, float64, full, int16, int32, int64, uint32
from numpy.typing import NDArray
from pypgtable.typing import SchemaColumn
from pypgtable.validators import PYPGTABLE_COLUMN_CONFIG_SCHEMA

from .gene_pool_common import GP_RAW_TABLE_SCHEMA

# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)

_REGEX: Literal['([A-Z]*)\\[{0,1}(\\d{0,})\\]{0,1}'] = r'([A-Z]*)\[{0,1}(\d{0,})\]{0,1}'


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
with open(join(dirname(__file__), "formats/gpc_field_definition_format.json"), "r", encoding="utf8") as file_ptr:
    merge(GPC_FIELD_SCHEMA, load(file_ptr))
gpc_field_validator: base_validator = base_validator(GPC_FIELD_SCHEMA, purge_uknown=True)
GPC_RAW_TABLE_SCHEMA: dict[str, dict[str, Any]] = deepcopy(GP_RAW_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gpc_table_format.json"), "r", encoding="utf8") as file_ptr:
    merge(GPC_RAW_TABLE_SCHEMA, load(file_ptr))
GPC_TABLE_SCHEMA: dict[str, ConfigDefinition] = {k: gpc_field_validator.normalized(v) for k, v in GPC_RAW_TABLE_SCHEMA.items()}
GPC_HIGHER_LAYER_COLS: tuple[str, ...] = tuple((key for key in filter(lambda x: x[0] == '_', GPC_TABLE_SCHEMA)))
GPC_UPDATE_RETURNING_COLS: tuple[str, ...] = tuple((x[1:] for x in GPC_HIGHER_LAYER_COLS)) + ('updated', 'created')
GPC_REFERENCE_COLUMNS: tuple[str, ...] = tuple((key for key, _ in filter(lambda x: x[1].get('reference', False), GPC_TABLE_SCHEMA.items())))
_GGC_INIT_LAMBDA: Callable[..., bool] = lambda x: not x[1].get('init_only', False) and not x[1].get('pgc_only', False)
_PGC_INIT_LAMBDA: Callable[..., bool] = lambda x: not x[1].get('init_only', False) and not x[1].get('ggc_only', False)

# Lazy add igraph
sql_np_mapping: dict[str, Any] = {
    'BIGINT': int64,
    'BIGSERIAL': int64,
    'BOOLEAN': bool_,
    'DOUBLE PRECISION': float64,
    'FLOAT8': float64,
    'FLOAT4': float32,
    'INT': int32,
    'INT8': int64,
    'INT4': int32,
    'INT2': int16,
    'INTEGER': int32,
    'REAL': float32,
    'SERIAL': int32,
    'SERIAL8': int64,
    'SERIAL4': int32,
    'SERIAL2': int16,
    'SMALLINT': int16,
    'SMALLSERIAL': int16
}

_MODIFIED: Field = {
    'type': bool,
    'length': 1,
    'default': False,
    'read_only': False,
    'read_count': 0,
    'write_count': 0
}


def create_cache_config(table_format_json: dict[str, ConfigDefinition]) -> dict[str, Field]:
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
        match: Match[str] | None = search(_REGEX, definition['type'])
        assert match is not None, f"definition['type'] = {definition['type']} which cannt be parsed!"
        typ: str = 'LIST'
        if match is not None and '[]' not in definition['type']:
            typ = match.group(1)
        length: int = 1 if not match.group(2) else int(match.group(2))
        fields[column] = {
            'type': sql_np_mapping.get(typ, list) if not definition.get('indexed', False) else _indexed_store,
            'length': length,
            'default': sql_np_mapping[typ](0) if typ in sql_np_mapping else None,
            'read_only': not definition.get('volatile', False),
            'read_count': 0,
            'write_count': 0
        }
    return fields


# The Gene Pool Cache (GPC) has a dictionary like interface plus some additional
# access functions for better performance of common operations. An actual dictionary
# uses way too many resources but is easier to implement.
# In the short term (and may be in the long term as a validation reference)
# the GPC is implemented as a dictionary and the 'optimised' access
# functions emulated.


def next_idx_generator(delta_size: int, empty_list: list[int], allocate_func: Callable[[], None]) -> Generator[int, None, None]:
    """Generate the next valid storage idx.

    Storage indicies start at 0 and increment to infinity.
    However the next available index may not be n+1.
    Deleted entries indices are added to the empty_list and are prioritised
    for new storage.
    If all pre-allocated storage is consumed the storage will be expanded by
    delta_size indices.

    Args
    ----
    delta_size: The log2(number of entries) to increase storage capacity by when all existing storage is occupied.
    empty_list: The indices of empty entries in the data store.
    allocate_func: A function that increases the number of entries by delta_size (takes no parameters)

    Returns
    -------
    A next index generator.
    """
    allocation_round: int = 0
    allocation_base: int = 0
    while True:
        _logger.debug(f"Creating new allocation {allocation_round} at base {allocation_base:08x}.")
        allocate_func()
        allocation_base = 2**delta_size * allocation_round
        for idx in range(allocation_base, allocation_base + 2**delta_size):
            while empty_list:
                if _LOG_DEBUG:
                    _logger.debug(f"Using deleted entry index {empty_list[0]} for next new entry.")
                yield empty_list.pop(0)
            yield idx
        allocation_round += 1


def allocate(data: dict[str, list[Any]], delta_size: int, fields: dict[str, Field]) -> None:
    """Allocate storage space for data.

    Each field has a particular storage type for its needs. The only requirement
    that it supports __getitem__() & __setitem__().

    Args
    ----
    data: Data store to expand.
    delta_size: The log2(number of entries) to increase storage capacity by when all existing storage is occupied.
    fields: Definition of data fields
    """
    # This is ordered so read-only fields are allocated together for CoW performance.
    size: int = 2**delta_size
    indexed_stores: dict[str, _indexed_store] = {}
    for key, value in sorted(fields.items(), key=lambda x: x[1].get('read_only', True)):
        shape: int | tuple[int, int] = size if value.get('length', 1) == 1 else (size, value['length'])
        if isinstance(value['type'], list):
            data[key].append([value['default']] * size)
        elif not isinstance(value['type'], _indexed_store):
            data[key].append(full(shape, value['default'], dtype=value['type']))
        else:
            istore: _indexed_store = indexed_stores[key] if key in indexed_stores else _indexed_store(size)
            data[key].append(istore.allocate())


class devnull():
    """Behaves like Linux /dev/null."""

    def __len__(self) -> Literal[0]:
        return 0

    def __getitem__(self, k) -> Self:
        return self

    def __setitem__(self, k, v) -> None:
        pass


class _store():
    """Store data compactly with a dict like interface."""

    def __init__(self, fields: dict[str, Field] | None = None, delta_size: int = 16) -> None:
        """Create a _gpc object.

        _gpc data is stored in numpy arrays or list if not numeric.

        Args
        ----
        fields: Describes the storage type and property hints and must have the following format:
            {
                <field name>: {
                    'type': Any         # A valid numpy type or None
                    'default': Any      # The value the property takes if not specified.
                    'read-only': bool   # True if read-only
                    'length': int       # The number of elements in an array type. 1 for scalar fields.
                },
                ...
            }
        delta_size: The log2(minimum number) of entries to add to the allocation when space runs out.
        """
        if fields is None:
            fields = {}
        self.fields: dict[str, Field] = deepcopy(fields)
        self.fields['__modified__'] = deepcopy(_MODIFIED)
        self.writable_fields: dict[str, Field] = {k: v for k, v in self.fields.items() if not v['read_only']}
        self.read_only_fields: dict[str, Field] = {k: v for k, v in self.fields.items() if v['read_only']}
        self.delta_size: int = delta_size
        self.ref_to_idx: dict[int, int] = {}
        self._data: dict[str, list[Any]] = {k: [] for k in self.fields.keys()}
        self._devnull: devnull = devnull()
        self._idx_mask: int = (1 << self.delta_size) - 1
        _logger.debug(f"Fields created: {tuple(self._data.keys())}")
        self._empty_list: list[int] = []
        _allocate: partial[None] = partial(allocate, data=self._data, delta_size=delta_size, fields=self.fields)
        self._idx: Generator[int, None, None] = next_idx_generator(delta_size, self._empty_list, _allocate)
        next(self._idx)  # Allocation 0, index 0 = None

    def __del__(self) -> None:
        """Check to see if _store has been reasonably utilised."""
        if __debug__:
            for field, value in self.fields.items():
                if not value['read_count'] + value['write_count']:
                    _logger.warning(f"'{field}' was neither read nor written!")
                if not value['read_only'] and not value['write_count']:
                    _logger.warning(f"'{field}' is writable but was never written!")

    def __contains__(self, ref: int) -> bool:
        """Test if a ref is in the GPC.

        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        True if ref is present.
        """
        # Reference 0 = None which is not in the GPC
        return ref in self.ref_to_idx if ref else False

    def __delitem__(self, ref: int) -> None:
        """Remove the entry for ref from the GPC.

        Args
        ----
        ref: GPC unique GC reference.
        """
        # Reference 0 is None which is not in the GPC
        if not ref:
            raise KeyError("Cannot delete the 0 (null) reference).")
        full_idx: int = self.ref_to_idx[ref]
        del self.ref_to_idx[ref]

        self._empty_list.append(full_idx)
        allocation: int = full_idx >> self.delta_size
        idx: int = full_idx & self._idx_mask
        if _LOG_DEBUG:
            _logger.debug(f"Deleting ref {ref_str(ref)}: Allocation {allocation} index {idx}.")
        for k, v in self.fields.items():
            self._data[k][allocation][idx] = v['default']

    def __getitem__(self, ref: int) -> xGC:
        """Return an gGC dict-like structure from the GPC.

        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        dict-like GC record.
        """
        # Reference 0 is None which is not in the GPC
        if not ref:
            raise KeyError("Cannot read the 0 (null) reference).")
        full_idx: int = self.ref_to_idx[ref]
        allocation: int = full_idx >> self.delta_size
        idx: int = full_idx & self._idx_mask
        if _LOG_DEBUG:
            _logger.debug(f"Getting GGC ref {ref_str(self._data['ref'][allocation][idx])}"
                          f" from allocation {allocation}, index {idx} (full index = {full_idx:08x}).")
        return xGC(self._data, allocation, idx, self.fields)

    def __len__(self) -> int:
        """The number of entries."""
        return len(self.ref_to_idx)

    def __setitem__(self, ref: int, value: dict | eGC | mGC | gGC) -> None:
        """Create a gGC entry in the GPC.

        NOTE: Set of an existing entry behaves like an update()

        Args
        ----
        ref: The GC unique reference in the GP.
        value: A dict-like object defining at least the required fields of a gGC.
        """
        if not ref:
            raise KeyError("Cannot set the 0 (null) reference.")
        full_idx: int | None = self.ref_to_idx.get(ref)
        if full_idx is None:
            modified: bool = False
            full_idx = next(self._idx)
            self.ref_to_idx[ref] = full_idx
            if _LOG_DEBUG:
                _logger.debug("Ref does not exist in the GPC generating full index.")
        else:
            modified = True
        allocation: int = full_idx >> self.delta_size
        idx: int = full_idx & self._idx_mask
        if _LOG_DEBUG:
            _logger.debug(f"Setting GGC ref {ref_str(ref)} to allocation {allocation},"
                          f" index {idx} (full index = {full_idx:08x}).")
        self._data['__modified__'][allocation][idx] = modified
        for k, v in value.items():
            # None means not set i.e. allocation default.
            if v is not None:
                self._data.get(k, self._devnull)[allocation][idx] = v
                if _LOG_DEBUG and k in self.fields:
                    self.fields[k]['write_count'] += 1
                    _logger.debug(f"Setting GGC key '{k}' to {self._data.get(k, self._devnull)[allocation][idx]}).")

    def __copy__(self) -> NoReturn:
        """Make sure we do not copy the GPC."""
        assert False, "Shallow copy of GPC."

    def __deepcopy__(self, obj: Any) -> NoReturn:
        """Make sure we do not copy GPC."""
        assert False, "Deep copy of the GPC."

    def keys(self) -> dict_keys[int]:
        """A view of the references in the GPC."""
        return self.ref_to_idx.keys()

    def items(self) -> Generator[tuple[int, xGC], None, None]:
        """A view of the gGCs in the GPC."""
        for ref in self.ref_to_idx:
            yield ref, self[ref]

    def update(self, value) -> None:
        """Update the GPC with a dict-like collection of gGCs."""
        for k, v in value.items():
            self[k] = v

    def values(self) -> Generator[xGC, None, None]:
        """A view of the gGCs in the GPC."""
        for ref in self.ref_to_idx:
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
        fields: dict[str, Field] = self.fields if all_fields else self.writable_fields
        for full_idx in self.ref_to_idx.values():
            allocation: int = full_idx >> self.delta_size
            idx: int = full_idx & self._idx_mask
            if self._data['__modified__'][allocation][idx]:
                yield xGC(self._data, allocation, idx, fields)


class _indexed_store_allocation():

    def __init__(self, store: _indexed_store) -> None:
        self._data: list[Any] = store._data
        self._empty_set: set[uint32] = store._empty_set
        self._allocation: NDArray[uint32] = full(store._size, 0, dtype=uint32)

    def __getitem__(self, idx) -> Any:
        return self._data[self._allocation[idx]]

    def __setitem__(self, idx, value) -> None:
        if not self._empty_set:
            empty_idx: uint32 = self._empty_set.pop()
            self._allocation[idx] = empty_idx
            self._data[empty_idx] = value
        else:
            self._allocation[idx] = len(self._data)
            self._data.append(value)

    def __delitem__(self, idx: int) -> None:
        deleted_idx: uint32 = self._allocation[idx]
        self._allocation[idx] = 0
        if deleted_idx:
            self._data[deleted_idx] = None
            self._empty_set.add(deleted_idx)


class _indexed_store():
    """For sparsely populated fields to reduce storage costs.

    Becomes economic when 'fraction populated' * 'allocation size' > 4

    """

    def __init__(self, size) -> None:
        self._data: list[Any] = [None]
        self._size = size
        self._empty_set: set[uint32] = set()
        self._num_allocations: int = 0

    def allocate(self) -> _indexed_store_allocation:
        """Create a new allocation."""
        self._num_allocations += 1
        return _indexed_store_allocation(self)

    def __del__(self) -> None:
        _logger.debug(f"{len(self._data)} elements in indexed store versus {self._size * self._num_allocations} allocated.")


class gene_pool_cache():
    """The Gene Pool Cache (GPC)."""

    def __init__(self, delta_size: int = 17) -> None:
        fields: dict[str, ConfigDefinition] = {k: v for k, v in filter(_GGC_INIT_LAMBDA, GPC_TABLE_SCHEMA.items())}
        self._ggc_cache: _store = _store(create_cache_config(fields), delta_size)
        self._ggc_refs: dict[int, int] = self._ggc_cache.ref_to_idx
        fields: dict[str, ConfigDefinition] = {k: v for k, v in filter(_PGC_INIT_LAMBDA, GPC_TABLE_SCHEMA.items())}
        self._pgc_cache: _store = _store(create_cache_config(fields), delta_size)
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

        Args
        ----
        ref: GPC unique GC reference.
        """
        if ref in self._ggc_refs:
            del self._ggc_cache[ref]
        else:
            del self._pgc_cache[ref]

    def __getitem__(self, ref: int) -> xGC:
        """Return an gGC dict-like structure from the GPC.

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

    def __len__(self) -> int:
        """The number of entries."""
        return len(self._ggc_refs) + len(self._pgc_refs)

    def __setitem__(self, ref: int, value: dict | eGC | mGC | gGC) -> None:
        """Create a gGC entry in the GPC.

        NOTE: Set of an existing entry behaves like an update()

        Args
        ----
        ref: The GC unique reference in the GP.
        value: A dict-like object defining at least the required fields of a gGC.
        """
        if is_pgc(value):
            self._pgc_cache[ref] = value
        else:
            self._ggc_cache[ref] = value

    def __copy__(self) -> NoReturn:
        """Make sure we do not copy the GPC."""
        assert False, "Shallow copy of GPC."

    def __deepcopy__(self, obj: Any) -> NoReturn:
        """Make sure we do not copy GPC."""
        assert False, "Deep copy of the GPC."

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
