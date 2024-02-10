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
from gc import collect
from typing import Any, overload
from logging import Logger, getLogger, NullHandler
from numpy import empty, int64, argsort, bytes_, int32, full, zeros
from numpy.typing import NDArray

from egp_types.store import static_store, dynamic_store, DDSL
from egp_types._genetic_code import _genetic_code, PURGED_GENETIC_CODE, EMPTY_GENETIC_CODE, EMPTY_GC_IDX, PURGED_GC_IDX
from egp_types.graph import graph
from .genomic_library import genomic_library


# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


# Constants
GPC_DEFAULT_SIZE: int = 2**10


class gc_index():
    """A genetic code index."""
    indices: NDArray[int32]
    genetic_codes: NDArray[Any]

    def __delitem__(self, idx: int) -> None:
        """Free the specified index by pointing to the EMPTY_GENETIC_CODE."""
        cls = type(self)
        cls.indices[idx] = 0

    def __getitem__(self, idx: int) -> Any:
        """Return the genetic code at the specified index."""
        cls = type(self)
        return cls.genetic_codes[cls.indices[idx]]

    def __setitem__(self, idx: int, val: int32 | int) -> None:
        """Set the genetic code at the specified index."""
        cls = type(self)
        cls.indices[idx] = val


class gca(gc_index):
    """Genetic Code A index."""

class gcb(gc_index):
    """Genetic Code B index."""

class ancestor_a(gc_index):
    """Ancestor A index."""

class ancestor_b(gc_index):
    """Ancestor B index."""

class gpc_ds_common(static_store):
    """Gene Pool Cache dynamic store common to all genetic code types."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the storage."""
        super().__init__(*args, **kwargs)
        self.signature: NDArray[bytes_] = empty((self._size, 32), dtype=bytes_)


class ds_index_wrapper():
    """Wrapper for dynamic store index."""

    _index_mapping: NDArray[int32]
    _dynamic_store: dynamic_store
    _genetic_codes: NDArray[Any]

    def __init__(self, member: str) -> None:
        """Initialize the wrapper."""
        self.member: str = member

    def __delitem__(self, _: int) -> None:
        """Removing a member element is not supported."""
        raise RuntimeError("The dynamic store does not support deleting member elements.")

    def __getitem__(self, idx: int) -> Any:
        """Return the object at the specified index."""
        cls = type(self)
        mapping_idx: int32 = cls._index_mapping[idx]
        if mapping_idx == -1:
            # If there is no mapping then the attribute is dynamically calculated
            return getattr(cls._genetic_codes[idx], self.member)()
        return cls._dynamic_store[self.member][mapping_idx]

    def __setitem__(self, idx: int, val: Any) -> None:
        """Set the object at the specified index."""
        cls = type(self)
        mapping_idx: int32 | int = cls._index_mapping[idx]
        # If the mapping has not yet been created then create it.
        if mapping_idx == -1:
            mapping_idx = cls._dynamic_store.next_index()
            cls._index_mapping[idx] = mapping_idx
        cls._dynamic_store[self.member][mapping_idx] = val


class common_ds_index_wrapper(ds_index_wrapper):
    """Wrapper for common dynamic store index."""


class gene_pool_cache(static_store):
    """A memory efficient store genetic codes."""

    # The global genomic library
    # Is used in nodes for lazy loading of dependent nodes
    gl: genomic_library = genomic_library()

    def __init__(self, size: int = GPC_DEFAULT_SIZE) -> None:
        """Initialize the storage."""
        super().__init__(size)
        # Static store members
        self.genetic_code: NDArray[Any] = empty(self._size, dtype=_genetic_code)
        self.gca: gc_index = gca()
        gca.indices = zeros(self._size, dtype=int32)
        gca.genetic_codes = self.genetic_code
        self.gcb: gc_index = gcb()
        gcb.indices = zeros(self._size, dtype=int32)
        gcb.genetic_codes = self.genetic_code
        self.graph: NDArray[Any] = empty(self._size, dtype=graph)
        self.ancestor_a: gc_index = ancestor_a()
        ancestor_a.indices = zeros(self._size, dtype=int32)
        ancestor_a.genetic_codes = self.genetic_code
        self.ancestor_b: gc_index = ancestor_b()
        ancestor_b.indices = zeros(self._size, dtype=int32)
        ancestor_b.genetic_codes = self.genetic_code

        # Utility static store members
        # Access sequence of genetic codes. Used to determine which ones were least recently used.
        self.access_sequence: NDArray[int64] = empty(self._size, dtype=int64)
        # Common dynamic store indices. -1 means not in the common dynamic store.
        self.common_ds_idx: NDArray[int32] = full(self._size, int32(-1), dtype=int32)

        # Not static store members: Must begin with '_'
        self._common_ds = dynamic_store(gpc_ds_common, max((size.bit_length() - 4, DDSL)))

        # Set up dynamic store member index wrappers
        common_ds_index_wrapper._index_mapping = self.common_ds_idx
        common_ds_index_wrapper._dynamic_store = self._common_ds
        common_ds_index_wrapper._genetic_codes = self.genetic_code
        self._common_ds_members: dict[str, common_ds_index_wrapper] = {m: common_ds_index_wrapper(m) for m in self._common_ds.members}

        # Initialise the genetic code index
        self.genetic_code[EMPTY_GC_IDX] = EMPTY_GENETIC_CODE
        self.genetic_code[PURGED_GC_IDX] = PURGED_GENETIC_CODE

    @overload
    def __getitem__(self, item: int) -> _genetic_code: ...

    @overload
    def __getitem__(self, item: str) -> Any: ...

    def __getitem__(self, item) -> Any:
        """Return the object at the specified index or the member to be indexed.
        There are 3 possible look up methods:
        1. By index - return the genetic code at the index
        2. By static store member name - return the member from the static store which then can be indexed
        3. By dynamic store member name - return a wrapper to map the GPC index to the dynamic store index     
        """
        if isinstance(item, int):
            return self.genetic_code[item]
        # First see if a member is in the static store
        member: Any = getattr(gene_pool_cache, item, None)
        return member if member is not None else self._common_ds_members[item]

    def __delitem__(self, idx: int) -> None:
        """Free the specified index. Note this does not try and remove all references as purge() does."""
        #TODO: Push to genomic library
        self.genetic_code[idx] = PURGED_GENETIC_CODE
        self.gca[idx] = 0
        self.gcb[idx] = 0
        self.graph[idx] = None
        self.ancestor_a[idx] = 0
        self.ancestor_b[idx] = 0
        super().__delitem__(idx)
        if self.common_ds_idx[idx] != -1:
            del self._common_ds[self.common_ds_idx[idx]]
            self.common_ds_idx[idx] = -1

    def assign_index(self, obj: _genetic_code) -> int:
        """Return the next index for a new node."""
        idx: int = self.next_index()
        self.genetic_code[idx] = obj
        return idx

    def next_idx(self) -> int:
        """Return the next available index. If there are no more purge the genetic codes that have not been
        used in the longest time."""
        try:
            idx: int = self.next_index()
        except OverflowError:
            self.purge()
            idx = self.next_index()
        return idx

    def reset(self, size: int | None = None) -> None:
        """A full reset of the store allows the size to be changed. All genetic codes
        are deleted which pushes the genetic codes to the genomic library as required."""
        super().reset(size)
        # Static store members
        self.genetic_code: NDArray[Any] = empty(self._size, dtype=_genetic_code)
        self.gca: gc_index = gca()
        gca.indices = zeros(self._size, dtype=int32)
        gca.genetic_codes = self.genetic_code
        self.gcb: gc_index = gcb()
        gcb.indices = zeros(self._size, dtype=int32)
        gcb.genetic_codes = self.genetic_code
        self.graph: NDArray[Any] = empty(self._size, dtype=graph)
        self.ancestor_a: gc_index = ancestor_a()
        ancestor_a.indices = zeros(self._size, dtype=int32)
        ancestor_a.genetic_codes = self.genetic_code
        self.ancestor_b: gc_index = ancestor_b()
        ancestor_b.indices = zeros(self._size, dtype=int32)
        ancestor_b.genetic_codes = self.genetic_code

        # Utility static store members
        # Access sequence of genetic codes. Used to determine which ones were least recently used.
        self.access_sequence: NDArray[int64] = empty(self._size, dtype=int64)
        # Common dynamic store indices. -1 means not in the common dynamic store.
        self.common_ds_idx: NDArray[int32] = full(self._size, int32(-1), dtype=int32)

        # Initialise the genetic code index
        self.genetic_code[EMPTY_GC_IDX] = EMPTY_GENETIC_CODE
        self.genetic_code[PURGED_GC_IDX] = PURGED_GENETIC_CODE

        # Clean up the heap
        _logger.debug(f"{collect()} unreachable objects not collected after purge.")

        # Total = 2* 8 + 5 * 4 = 36 bytes + base class per element

    def purge(self, fraction: float = 0.25) -> None:
        """Purge the store of unused data."""
        # Simply marking the data as unused is insufficient because the purged
        # data may be referenced by other objects. The purge function ensures that
        # all references to the purged data in the store are removed.
        num_to_purge: int = int(self._size * fraction)
        _logger.info(f"Purging {int(100*fraction)}% = ({num_to_purge} of {self._size}) of the store")
        purge_indices: set[int] = set(argsort(self.access_sequence)[:num_to_purge])

        # Convert GC's with purged dependents into leaf nodes
        for gc in self.genetic_code:
            gc.make_leaf(purge_indices)

        # Delete the purged objects
        for idx in purge_indices:
            del self[idx]


        # Clean up the heap
        _logger.debug(f"{collect()} unreachable objects not collected after purge.")


# Instanciate the gene pool cache
_genetic_code.gene_pool_cache = gene_pool_cache()
