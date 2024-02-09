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
from numpy import empty, int64, argsort
from numpy.typing import NDArray

from egp_types.store import static_store, dynamic_store
from egp_types._genetic_code import _genetic_code, PURGED_GENETIC_CODE
from egp_types.graph import graph
from .genomic_library import genomic_library


# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


# Constants
GPC_DEFAULT_SIZE: int = 2**10


class gene_pool_cache(static_store):
    """A memory efficient store genetic codes."""

    # The global genomic library
    # Is used in nodes for lazy loading of dependent nodes
    gl: genomic_library = genomic_library()

    def __init__(self, size: int = GPC_DEFAULT_SIZE) -> None:
        """Initialize the storage."""
        super().__init__(size, PURGED_GENETIC_CODE, lambda x: None)
        self.genetic_code: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.gca: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.gcb: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.graph: NDArray[Any] = empty(self.size, dtype=graph)
        self.ancestor_a: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.ancestor_b: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.descendants: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.access_sequence: NDArray[int64] = empty(self.size, dtype=int64)
        self._dynamic_store = dynamic_store(self.size, PURGED_GENETIC_CODE)

    @overload
    def __getitem__(self, item: int) -> _genetic_code: ...

    @overload
    def __getitem__(self, item: str) -> Any: ...

    def __getitem__(self, item) -> Any:
        """Return the object at the specified index or the member to be indexed."""
        if isinstance(item, int):
            return self.genetic_code[item]
        # If looking for a member it may be in the static store or the dynamic store
        member: Any = getattr(gene_pool_cache, item, None)
        if member is not None:
            return member
        return self._dynamic_store[item]

    def __delitem__(self, idx: int) -> None:
        """Remove the object at the specified index."""
        #TODO: Push to genomic library
        self.genetic_code[idx] = self.purged_object
        self.empty_indices.append(idx)

    def assign_index(self, obj: _genetic_code) -> int:
        """Return the next index for a new node."""
        idx: int = self.next_index()
        self.genetic_code[idx] = obj
        return idx

    def reset(self, size: int | None = None) -> None:
        """A full reset of the store allows the size to be changed. All genetic codes
        are deleted which pushes the genetic codes to the genomic library as required."""
        super().reset(size)
        self.genetic_code: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.gca: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.gcb: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.graph: NDArray[Any] = empty(self.size, dtype=graph)
        self.ancestor_a: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.ancestor_b: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.descendants: NDArray[Any] = empty(self.size, dtype=_genetic_code)
        self.access_sequence: NDArray[int64] = empty(self.size, dtype=int64)
        # Total = 8 * 8 = 64 bytes + base class per element

    def _purge(self, fraction: float = 0.25) -> list[int]:
        """Purge the store of unused data."""
        # Simply marking the data as unused is insufficient because the purged
        # data may be referenced by other objects. The purge function ensures that
        # all references to the purged data in the store are removed.
        assert not self.empty_indices, "empty_indices is not empty"
        num_to_purge: int = int(self.size * fraction)
        _logger.info(f"Purging {int(100*fraction)}% = ({num_to_purge} of {self.size}) of the store")
        purge_indices: list[int] = argsort(self.access_sequence)[:num_to_purge].tolist()

        # Do what is necessary with the doomed objects before they are purged
        self.purge(purge_indices)

        # Remove any references to the purged objects
        for idx in purge_indices:
            del self[idx]

        # Clean up the heap
        _logger.debug(f"{collect()} unreachable objects not collected after purge.")
        return purge_indices


# Instanciate the gene pool cache
_genetic_code.gene_pool_cache = gene_pool_cache()
