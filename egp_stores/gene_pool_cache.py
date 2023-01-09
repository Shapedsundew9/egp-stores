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

For read-only GC's in the persistent Gene Pool loaded on startup ~75% of the data
is read only avoiding 4x as many CoW's giving a total factor of ~16x for that data.
Bit of an anti-pattern for python but in this case the savings are worth it.
"""

from logging import DEBUG, NullHandler, getLogger
from typing import Any, Dict, List, Callable, Generator
from functools import partial
from egp_types.eGC import eGC
from egp_types.mGC import mGC
from egp_types.GGC import GGC
from egp_types.gc_type_tools import merge, ref_str, is_pgc
from copy import deepcopy
from numpy import bool_, int64, int32, int16, float32, float64, full
from re import search
from .gene_pool_table_schema import GP_TABLE_SCHEMA
from json import load
from os.path import dirname, join

# Logging
_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)
_SUPPRESS_VALIDATION = not _LOG_DEBUG 

# If this field exists and is not None the gGC is a pGC
_PROOF_OF_PGC = 'pgc_fitness'
_REGEX = r'([A-Z]*)\[{0,1}(\d{0,})\]{0,1}'

# Load the GPC in memory schema
GPC_TABLE_SCHEMA = deepcopy(GP_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gpc_table_format.json"), "r") as file_ptr:
    merge(GPC_TABLE_SCHEMA, load(file_ptr))

# Lazy add igraph
sql_np_mapping = {
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


def _test_helper():
    """Needed to suppress deep validation of GGC() initialization when testing."""
    global _SUPPRESS_VALIDATION
    _SUPPRESS_VALIDATION = True


def create_cache_config(table_format_json: Dict[str, Dict[str, str|int|float]]) -> Dict[str, Dict[str, Any]]:
    """Converts the Gene Pool table format to a GPC config.
    
    Args
    ----
    table_format_json: The normalized GP pypgtable format JSON

    Returns
    -------
    See _gpc class fields definition.
    """
    # TODO: Add classes to lookup common large collective fields like input*, output*
    # Thinking something like a hash stored per entry that can be used to look up the
    # specifics. Monitor the unique hash to entry ratio to make sure it is worth it.
    cconfig = {}
    for column, definition in table_format_json.items():
        m = search(_REGEX, definition['type'])
        typ = 'LIST'
        if m is not None and '[]' not in definition['type']:
            typ = m.group(1)
        length = 1 if not len(m.group(2)) else int(m.group(2))
        cconfig[column] = {
            'type': sql_np_mapping.get(typ, list),
            'length': length,
            'default': sql_np_mapping[typ](0) if typ in sql_np_mapping else None,
            'read_only': not definition.get('volatile', False)
        }
    return cconfig 


# The Gene Pool Cache (GPC) has a dictionary like interface plus some additional
# access functions for better performance of common operations. An actual dictionary
# uses way too many resources but is easier to implement.
# In the short term (and may be in the long term as a validation reference)
# the GPC is implemented as a dictionary and the 'optimised' access
# functions emulated.

class xGC():

    def __init__(self, _data:Any, allocation:int, idx:int, fields:Dict) -> None:
        self._data = _data
        self._allocation = allocation
        self._idx = idx
        self._fields = fields

    def __contains__(self, key:str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        if _LOG_DEBUG:
            assert key in self._data, f"{key} is not a key in data. Are you trying to get a pGC field from a gGC?"
            self._fields[key]['read_count'] += 1
            _logger.debug(f"Getting GGC key '{key}' from allocation {self._allocation}, index {self._idx}).")
        return self._data[key][self._allocation][self._idx]

    def __setitem__(self, key: str, value: Any) -> None:
        if _LOG_DEBUG:
            assert key in self._data, f"'{key}' is not a key in data. Are you trying to set a pGC field in a gGC?"
            assert not self._fields[key]['read_only'], f"Writing to read-only field '{key}'."
            self._fields[key]['write_count'] += 1
            _logger.debug(f"Setting GGC key '{key}' to allocation {self._allocation}, index {self._idx}).")
        self._data[key][self._allocation][self._idx] = value

    def __copy__(self):
        """Make sure we do not copy gGCs."""
        assert False, f"Shallow copy of xGC ref {self['ref']:016X}."

    def __deepcopy__(self):
        """Make sure we do not copy gGCs."""
        assert False, f"Deep copy of xGC ref {self['ref']:016X}."

    def keys(self):
        """A view of the keys in the xGC."""
        return self._data.keys()

    def is_pGC(self):
        """True if the xGC is a pGC."""
        return _PROOF_OF_PGC in self._data

    def items(self):
        """A view of the xGCs in the GPC."""
        for key in self._data.keys():
            yield key, self[key]

    def update(self, value):
        """Update the xGC with a dict-like collection of fields."""
        for k, v in value.items():
            self[k] = v

    def values(self):
        """A view of the field values in the xGC."""
        for key in self._data.keys():
            yield self[key]


def next_idx_generator(delta_size: int, empty_list: List[int], allocate_func: Callable[[], None]) -> Generator[int, None, None]:
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
    reallocate: A function that increases the number of entries by delta_size (takes no parameters)

    Returns
    -------
    A next index generator.
    """
    allocation_round = allocation_base = 0
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


def allocate(data:Dict[str, List[Any]], delta_size:int, fields:Dict[str, Dict[str, Any]]) -> None:
    """Allocate storage space for data.

    Args
    ----
    data: Data store to expand.
    delta_size: The log2(number of entries) to increase storage capacity by when all existing storage is occupied.
    fields: Definition of data fields
    """
    # This is ordered so read-only fields are allocated together for CoW performance.
    for key, value in sorted(fields.items(), key=lambda x: x[1].get('read_only', True)):
        shape = 2**delta_size if value.get('length', 1) == 1 else (2**delta_size, value['length'])
        if not isinstance(value['type'], list):
            data[key].append(full(shape, value['default'], dtype=value['type']))
        else:
            data[key].append([value['default']] * 2**delta_size)

class devnull():

    def __len__(self):
        return 0

    def __getitem__(self, k):
        return self

    def __setitem__(self, k, v):
        pass

class _gpc():
    """Mapping to a GC stored in the gene pool cache."""

    def __init__(self, fields: Dict[str, Dict[str, Any]] = {}, delta_size:int = 16) -> None:
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
        self.fields = deepcopy(fields)
        self.delta_size = delta_size
        self.ref_to_idx = {}
        self._data = {k: [] for k in self.fields.keys()}
        self._devnull = devnull()
        self._idx_mask = (1 << self.delta_size) - 1
        _logger.debug(f"Fields created: {tuple(self._data.keys())}")
        self._empty_list = []
        _allocate = partial(allocate, data=self._data, delta_size=delta_size, fields=self.fields)
        self._idx = next_idx_generator(delta_size, self._empty_list, _allocate)
        if _LOG_DEBUG:
            for value in self.fields.values():
                value['read_count'] = 0
                value['write_count'] = 0

    def __del__(self):
        """Check to see if data  store has been reasonably utilised."""
        if _LOG_DEBUG:
            for field, value in self.fields.items():
                if not (value['read_count'] + value['write_count']):
                    _logger.warning(f"'{field}' was neither read nor written!")
                if not value['read_only'] and not value['write_count']:
                    _logger.warning(f"'{field}' is writable but was never written!")

    def __contains__(self, ref:int) -> bool:
        """Test if a ref is in the GPC.
        
        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        True if ref is present.
        """
        return ref in self.ref_to_idx

    def __delitem__(self, ref:int) -> None:
        """Remove the entry for ref from the GPC.
        
        Args
        ----
        ref: GPC unique GC reference.
        """
        full_idx = self.ref_to_idx[ref]
        del self.ref_to_idx[ref]

        self._empty_list.append(full_idx)
        allocation = full_idx >> self.delta_size
        idx = full_idx & self._idx_mask
        if _LOG_DEBUG:
            _logger.debug(f"Deleting ref {ref_str(ref)}: Allocation {allocation} index {idx}.")
        for k, v in self.fields.items():
            self._data[k][allocation][idx] = v['default']

    def __getitem__(self, ref:int) -> xGC:
        """Return an gGC dict-like structure from the GPC.
        
        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        dict-like GC record.        
        """
        full_idx = self.ref_to_idx[ref]
        allocation = full_idx >> self.delta_size
        idx = full_idx & self._idx_mask
        if _LOG_DEBUG:
            _logger.debug(f"Getting GGC ref {ref_str(self._data['ref'][allocation][idx])}"
                f" from allocation {allocation}, index {idx} (full index = {full_idx:08x}).")
        return xGC(self._data, allocation, idx, self.fields)

    def __len__(self):
        """The number of entries."""
        return len(self.ref_to_idx)

    def __setitem__(self, ref:int, value: dict | eGC | mGC | GGC) -> None:
        """Create a gGC entry in the GPC.

        NOTE: Set of an existing entry behaves like an update()

        Args
        ----
        ref: The GC unique reference in the GP.
        value: A dict-like object defining at least the required fields of a gGC.
        """
        full_idx = self.ref_to_idx.get(ref)
        if full_idx is None:
            full_idx = next(self._idx)
            self.ref_to_idx[ref] = full_idx
            if _LOG_DEBUG:
                _logger.debug("Ref does not exist in the GPC generating full index.")
        allocation = full_idx >> self.delta_size
        idx = full_idx & self._idx_mask
        if _LOG_DEBUG:
            _logger.debug(f"Setting GGC ref {ref_str(ref)} to allocation {allocation},"
                f" index {idx} (full index = {full_idx:08x}).")
        for k, v in value.items():
            # None means not set i.e. allocation default.
            if v is not None:
                self._data.get(k, self._devnull)[allocation][idx] = v
                if _LOG_DEBUG:
                    self.fields[k]['write_count'] += 1
                    _logger.debug(f"Setting GGC key '{k}' to {self._data.get(k, self._devnull)[allocation][idx]}).")


    def __copy__(self):
        """Make sure we do not copy the GPC."""
        assert False, f"Shallow copy of GPC."

    def __deepcopy__(self):
        """Make sure we do not copy GPC."""
        assert False, f"Deep copy of the GPC."

    def keys(self):
        """A view of the references in the GPC."""
        return self.ref_to_idx.keys()

    def items(self):
        """A view of the gGCs in the GPC."""
        for ref in self.ref_to_idx.keys():
            yield ref, self[ref]

    def update(self, value):
        """Update the GPC with a dict-like collection of gGCs."""
        for k, v in value.items():
            self[k] = v

    def values(self):
        """A view of the gGCs in the GPC."""
        for ref in self.ref_to_idx.keys():
            yield self[ref]
        

class gene_pool_cache():
    
    def __init__(self, delta_size: int = 17) -> None:
        fields = filter(lambda x: not x[1].get('init_only', False) and not x[1].get('pgc_only', False), GPC_TABLE_SCHEMA.items())
        self._gGC_cache = _gpc(create_cache_config(dict(fields)), delta_size)
        self._gGC_refs = self._gGC_cache.ref_to_idx
        fields = filter(lambda x: not x[1].get('init_only', False) and not x[1].get('ggc_only', False), GPC_TABLE_SCHEMA.items())
        self._pGC_cache = _gpc(create_cache_config(dict(fields)), delta_size)
        self._pGC_refs = self._pGC_cache.ref_to_idx


    def __contains__(self, ref:int) -> bool:
        """Test if a ref is in the GPC.
        
        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        True if ref is present.
        """
        return ref in self._gGC_refs or ref in self._pGC_refs

    def __delitem__(self, ref:int) -> None:
        """Remove the entry for ref from the GPC.
        
        Args
        ----
        ref: GPC unique GC reference.
        """
        if ref in self._gGC_refs:
            del self._gGC_cache[ref]
        else:
            del self._pGC_cache[ref]

    def __getitem__(self, ref:int) -> xGC:
        """Return an gGC dict-like structure from the GPC.
        
        Args
        ----
        ref: GPC unique GC reference.

        Returns
        -------
        dict-like GC record.        
        """
        if ref in self._gGC_refs:
            return self._gGC_cache[ref]
        return self._pGC_cache[ref]

    def __len__(self):
        """The number of entries."""
        return len(self._gGC_refs) + len(self._pGC_refs)

    def __setitem__(self, ref:int, value: dict | eGC | mGC | GGC) -> None:
        """Create a gGC entry in the GPC.

        NOTE: Set of an existing entry behaves like an update()

        Args
        ----
        ref: The GC unique reference in the GP.
        value: A dict-like object defining at least the required fields of a gGC.
        """
        if is_pgc(value):
            self._pGC_cache[ref] = value
        else:
            self._gGC_cache[ref] = value

    def __copy__(self):
        """Make sure we do not copy the GPC."""
        assert False, f"Shallow copy of GPC."

    def __deepcopy__(self):
        """Make sure we do not copy GPC."""
        assert False, f"Deep copy of the GPC."

    def keys(self):
        """A view of the references in the GPC."""
        for key in self._gGC_refs.keys():
            yield key
        for key in self._pGC_refs.keys():
            yield key

    def is_pGC(self, ref):
        """True if the ref is that of a pGC."""
        return ref in self._pGC_refs

    def pGC_refs(self):
        """All pGC references in the GPC."""
        return list(self._pGC_refs.keys())

    def items(self):
        """A view of the gGCs in the GPC."""
        for ref in self._gGC_refs.keys():
            yield ref, self[ref]
        for ref in self._pGC_refs.keys():
            yield ref, self[ref]

    def update(self, value):
        """Update the GPC with a dict-like collection of gGCs."""
        for k, v in value.items():
            self[k] = v

    def values(self):
        """A view of the gGCs in the GPC."""
        for ref in self._gGC_refs.keys():
            yield self[ref]
        for ref in self._pGC_refs.keys():
            yield self[ref]
