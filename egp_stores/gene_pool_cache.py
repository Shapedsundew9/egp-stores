"""Gene Pool Cache.

The gene pool cache is a space and time optimised store of GC's. It is designed to
be multi-process friendly.

Naively, the gene pool cache could be implemented as a dictionary with reference keys.
This would be fast but does not scale well. Python dictionaries use huge amounts
of memory and are updated in a spatially broad manner requiring subprocesses to maintain
an almost full copy even if most entries are only read.

The gene pool cache as implemented here maintains a dictionary like interface but takes
advantage of some GC structural design choices to efficiently store data in numpy arrays where possible. 
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
from copy import deepcopy
from numpy import array


# Logging
_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)

# If this field exists and is not None the gGC is a pGC
_PROOF_OF_PGC = 'pgc_fitness'

# The Gene Pool Cache (GPC) has a dictionary like interface plus some additional
# access functions for better performance of common operations. An actual dictionary
# uses way too many resources but is easier to implement.
# In the short term (and may be in the long term as a validation reference)
# the GPC is implemented as a dictionary and the 'optimised' access
# functions emulated.

class xGC():

    def __init__(self, _data:Any, allocation:int, idx:int) -> None:
        self._data = _data
        self._allocation = allocation
        self._idx = idx

    def __contains__(self, key:str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        if _LOG_DEBUG:
            assert key in self._data, f"{key} is not a key in data. Are you trying to get a pGC field from a gGC?"
        return self._data[key][self._allocation][self._idx]

    def __setitem__(self, key: str, value: Any) -> None:
        if _LOG_DEBUG:
            assert key in self._data, f"{key} is not a key in data. Are you trying to set a pGC field in a gGC?"
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


def next_idx_generator(delta_size: int, empty_list: List[int], allocate_func: Callable[[], None]) -> Generator(int, None, None):
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
    allocation_round = 0
    while True:
        allocation_base = 2**delta_size * allocation_round
        for idx in range(allocation_base, allocation_base + 2**delta_size):
            while empty_list:
                yield empty_list.pop(0)
            yield idx
        allocation_round += 1
        allocate_func()


def allocate(data:Dict[str, List[Any]], delta_size:int, fields:Dict[str, Dict[str, Any]]) -> None:
    """Allocate storage space for data.

    Args
    ----
    data: Data store to expand.
    delta_size: The log2(number of entries) to increase storage capacity by when all existing storage is occupied.
    fields: Definition of data fields
    """
    # This is ordered so read-only fields are allocated together for CoW performance.
    for key, value in sorted(fields.items(), key=lambda x: x[1].get('read-only', True)):
        shape = delta_size if value.get('length', 1) else (2**delta_size, value['length'])
        if value['type'] is not None:
            data[key].append(array(shape, dtype=value['type'], fill_value=value['default']))
        else:
            data[key].append([value['default']] * 2**delta_size)


class _gpc():
    """Mapping to a GC stored in the gene pool cache."""

    def __init__(self, fields: Dict[str, Dict[str, Any]], delta_size:int = 16) -> None:
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
        self.ref_to_idx = {}
        self._data = {k: [] for k in self.fields.keys()}
        self._empty_list = []
        _allocate = partial(allocate, data=self._data, delta_size=delta_size, fields=self.fields)
        self._idx = next_idx_generator(delta_size, self._empty_list, _allocate)

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
        idx = full_idx & (self.delta_size - 1)
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
        idx = full_idx & (self.delta_size - 1)
        return xGC(self._data, allocation, idx)

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
        allocation = full_idx >> self.delta_size
        idx = full_idx & (self.delta_size - 1)
        if isinstance(value, dict | eGC | mGC):
            value = GGC(value)
            for k, v in value.items():
                self._data[k][allocation][idx] = v

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
        self._gGC_cache = _gpc(self.fields, delta_size)
        self._gGC_refs = self._gGC_cache.ref_to_idx
        self._pGC_cache = _gpc(self.fields, delta_size)
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

    def __setitem__(self, ref:int, value: dict | eGC | mGC | GGC) -> None:
        """Create a gGC entry in the GPC.

        NOTE: Set of an existing entry behaves like an update()

        Args
        ----
        ref: The GC unique reference in the GP.
        value: A dict-like object defining at least the required fields of a gGC.
        """
        if value.get(_PROOF_OF_PGC) is None:
            self.gGC_cache[ref] = value
        else:
            self.pGC_cache[ref] = value

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
