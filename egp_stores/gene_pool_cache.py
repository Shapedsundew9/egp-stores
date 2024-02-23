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
from gc import collect
from json import load
from logging import DEBUG, Logger, NullHandler, getLogger
from os.path import dirname, join
from typing import Any, Callable, Iterable, Iterator, cast

from egp_types._genetic_code import _genetic_code, EMPTY_GENETIC_CODE, NULL_SIGNATURE, STORE_PROXY_SIGNATURE_MEMBERS
from egp_types.connections import connections
from egp_types.genetic_code import genetic_code
from egp_types.graph import graph
from egp_types.interface import interface
from egp_types.rows import rows
from egp_utils.base_validator import base_validator
from egp_utils.common import merge
from egp_utils.store import DDSL, dynamic_store, static_store
from numpy import argsort, bytes_, empty, full, iinfo, int32, int64, zeros, ones, float32
from numpy.typing import NDArray
from pypgtable.pypgtable_typing import SchemaColumn
from pypgtable.validators import PYPGTABLE_COLUMN_CONFIG_SCHEMA

from .gene_pool_common import GP_RAW_TABLE_SCHEMA


# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)
_LOG_DEEP_DEBUG: bool = _logger.isEnabledFor(DEBUG - 1)


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


# Constants
GPC_DEFAULT_SIZE: int = 2**4
INT64_MAX: int = iinfo(int64).max


# Load the GPC config definition which is a superset of the pypgtable column definition
GPC_FIELD_SCHEMA: dict[str, dict[str, Any]] = deepcopy(PYPGTABLE_COLUMN_CONFIG_SCHEMA)
with open(join(dirname(__file__), "formats/gpc_field_definition_format.json"), "r", encoding="utf8") as file_ptr:
    merge(GPC_FIELD_SCHEMA, load(file_ptr))
gpc_field_validator: base_validator = base_validator(GPC_FIELD_SCHEMA, purge_uknown=True)
GPC_RAW_TABLE_SCHEMA: dict[str, dict[str, Any]] = deepcopy(GP_RAW_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gpc_table_format.json"), "r", encoding="utf8") as file_ptr:
    merge(GPC_RAW_TABLE_SCHEMA, load(file_ptr))
GPC_TABLE_SCHEMA: dict[str, ConfigDefinition] = {k: gpc_field_validator.normalized(v) for k, v in GPC_RAW_TABLE_SCHEMA.items()}
GPC_HIGHER_LAYER_COLS: tuple[str, ...] = tuple(key for key in filter(lambda x: x[0] == "_", GPC_TABLE_SCHEMA))
GPC_UPDATE_RETURNING_COLS: tuple[str, ...] = tuple(x[1:] for x in GPC_HIGHER_LAYER_COLS) + ("updated", "created")
GPC_REFERENCE_COLUMNS: tuple[str, ...] = tuple((key for key, _ in filter(lambda x: x[1].get("reference", False), GPC_TABLE_SCHEMA.items())))


class ds_index_wrapper:
    """Wrapper for dynamic store index."""

    dstore: dynamic_store
    genetic_codes: NDArray[Any]

    def __init__(self, member: str, index_mapping: NDArray[int32]) -> None:
        """Initialize the wrapper."""
        self.member: str = member
        self.index_mapping: NDArray[int32] = index_mapping

    def __delitem__(self, _: int) -> None:
        """Removing a member element is not supported. Delete the index in the store."""
        raise RuntimeError("The dynamic store does not support deleting member elements.")

    def __getitem__(self, idx: int) -> Any:
        """Return the object at the specified index."""
        cls = type(self)
        mapping_idx: int32 = self.index_mapping[idx]
        if mapping_idx == -1:
            # If there is no mapping then the attribute is dynamically calculated
            self.index_mapping[idx] = cls.dstore.next_index()
            getattr(cls.genetic_codes[idx], self.member)()
            mapping_idx = self.index_mapping[idx]
        return cls.dstore[self.member][mapping_idx]

    def __setitem__(self, idx: int, val: Any) -> None:
        """Set the object at the specified index."""
        cls = type(self)
        mapping_idx: int32 | int = self.index_mapping[idx]
        # If the mapping has not yet been created then create it.
        if mapping_idx == -1:
            mapping_idx = cls.dstore.next_index()
            self.index_mapping[idx] = mapping_idx
        cls.dstore[self.member][mapping_idx] = val


class common_ds_index_wrapper(ds_index_wrapper):
    """Wrapper for common dynamic store index."""


class gpc_ds_common(static_store):
    """Gene Pool Cache dynamic store for terminal genetic codes."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the storage."""
        super().__init__(*args, **kwargs)
        self.ancestor_a_signature: NDArray[bytes_] = zeros((self._size, 32), dtype=bytes_)
        self.ancestor_b_signature: NDArray[bytes_] = zeros((self._size, 32), dtype=bytes_)
        self.code_depth: NDArray[int32] = zeros(self._size, dtype=int32)
        self.codon_depth: NDArray[int32] = zeros(self._size, dtype=int32)
        self.gca_signature: NDArray[bytes_] = zeros((self._size, 32), dtype=bytes_)
        self.gcb_signature: NDArray[bytes_] = zeros((self._size, 32), dtype=bytes_)
        self.generation: NDArray[int64] = zeros(self._size, dtype=int64)
        self.num_codes: NDArray[int32] = zeros(self._size, dtype=int32)
        self.num_codons: NDArray[int64] = zeros(self._size, dtype=int64)
        self.pgc_signature: NDArray[bytes_] = zeros((self._size, 32), dtype=bytes_)
        self.signature: NDArray[bytes_] = zeros((self._size, 32), dtype=bytes_)
        # 224 bytes per entry (usually 8192 so 1835008) + 112 bytes per member (12 so 1324) + 56 bytes for the base class
        # Total = 1.8 MB per block

    def __delitem__(self, idx: int) -> None:
        """Free the specified index. Note this does not try and remove all references as purge() does."""
        self.ancestor_a_signature[idx] = NULL_SIGNATURE
        self.ancestor_b_signature[idx] = NULL_SIGNATURE
        self.code_depth[idx] = 0
        self.codon_depth[idx] = 0
        self.gca_signature[idx] = NULL_SIGNATURE
        self.gcb_signature[idx] = NULL_SIGNATURE
        self.num_codes[idx] = 0
        self.num_codons[idx] = 0
        self.generation[idx] = 0
        self.pgc_signature[idx] = NULL_SIGNATURE
        self.signature[idx] = NULL_SIGNATURE
        return super().__delitem__(idx)


class gene_pool_cache(static_store):
    """A memory efficient store genetic codes."""

    def __init__(self, size: int = GPC_DEFAULT_SIZE, push_to_gp: Callable[[tuple[genetic_code, ...]], None] = lambda x: None) -> None:
        """Initialize the storage."""
        super().__init__(size)
        # Static store members
        self.ancestor_a: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.ancestor_b: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.e_count: NDArray[int32] = ones(self._size, dtype=int32)
        self.evolvability: NDArray[float32] = ones(self._size, dtype=float32)
        self.f_count: NDArray[int32] = ones(self._size, dtype=int32)
        self.fitness: NDArray[float32] = zeros(self._size, dtype=float32)
        self.gca: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.gcb: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.graph: NDArray[Any] = empty(self._size, dtype=graph)
        self.pgc: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.properties: NDArray[int64] = zeros(self._size, dtype=int64)
        self.reference_count: NDArray[int64] = zeros(self._size, dtype=int64)
        self.survivability: NDArray[float32] = zeros(self._size, dtype=float32)
        # 84 bytes per entry (usually 2**20 so 88080384) 13 members at 112 bytes each = 1456 bytes + 56 bytes for the base class
        # Utility members below = 17 bytes = 17825792 bytes
        # Total = 101 MB + graphs
        # Graphs are hard to estimate: 384 in connections, 256 in rows + 64 in graph = 704 bytes per graph
        # Duplication rate of 8 (?) so 2**(20-3) * 704 = 88 MB
        # Assume dynamic store is 1/16 of the size of the static store so 1.8 * 2**(20-13-4) = 14.4 MB
        # Total of totals = 101 + 88 + 14.4 = 203.4 MB

        # Utility static store members
        # Access sequence of genetic codes. Used to determine which ones were least recently used.
        self.access_sequence: NDArray[int64] = full(self._size, INT64_MAX, dtype=int64)
        # The genetic codes themselves
        self.genetic_code: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        # Status byte for each genetic code.
        # 0 = dirty bit. If set then the genetic code has been modified and needs to be written to the GP.
        # 1:7 = reserved (read and written as 0)
        self.status_byte: NDArray[bytes_] = zeros(self._size, dtype=bytes_)

        # Common dynamic store indices. -1 means not in the common dynamic store.
        self.common_ds_idx: NDArray[int32] = full(self._size, int32(-1), dtype=int32)
        # Not static store members: Must begin with '_'
        self._common_ds = dynamic_store(gpc_ds_common, max((size.bit_length() - 7, DDSL)))
        # Set up dynamic store member index wrappers
        common_ds_index_wrapper.dstore = self._common_ds
        common_ds_index_wrapper.genetic_codes = self.genetic_code
        # If a member has the "_idx" suffix then it indexes the signatures store
        self._common_ds_members: dict[str, common_ds_index_wrapper] = {
            m: common_ds_index_wrapper(m, self.common_ds_idx) for m in self._common_ds.members
        }

        # Method to push genetic codes to the gene pool when the GPC is full
        self._push_to_gp: Callable[[tuple[genetic_code, ...]], None] = push_to_gp

    def __delitem__(self, idx: int) -> None:
        """Free the specified index. Note this does not try and remove all references as purge() does.
        It also does not push to the GP. It is intended to be used when the genetic code is no longer needed.
        """
        self.ancestor_a[idx] = EMPTY_GENETIC_CODE
        self.ancestor_b[idx] = EMPTY_GENETIC_CODE
        self.gca[idx] = EMPTY_GENETIC_CODE
        self.gcb[idx] = EMPTY_GENETIC_CODE
        self.graph[idx] = None
        self.pgc[idx] = EMPTY_GENETIC_CODE

        self.access_sequence[idx] = INT64_MAX
        self.genetic_code[idx] = EMPTY_GENETIC_CODE
        self.status_byte[idx] = 0
        super().__delitem__(idx)
        if self.common_ds_idx[idx] != -1:
            del self._common_ds[self.common_ds_idx[idx]]
            self.common_ds_idx[idx] = -1

    def __getitem__(self, idx: int) -> _genetic_code:
        """Return the object at the specified index or the member to be indexed.
        There are 3 possible look up methods:
        1. By index - return the genetic code at the index
        2. By static store member name - return the member from the static store which then can be indexed
        3. By dynamic store member name - return a wrapper to map the GPC index to the dynamic store index
        """
        if idx < 0:
            raise IndexError("Negative indices are not supported.")
        return self.genetic_code[idx]

    def __setitem__(self, _: str, __: Any) -> None:
        raise RuntimeError("The genetic code store does not support setting members directly. Use add().")

    def __iter__(self) -> Iterator[_genetic_code]:
        """Iterate over self."""
        return self.values()

    def assign_index(self, obj: _genetic_code) -> int:
        """Return the next index for a new genetic code. DO NOT USE outside of the
        gene_pool_cache or genetic_code classes. Use add() instead."""
        idx: int = self.next_index()
        self.genetic_code[idx] = obj
        return idx

    def next_index(self) -> int:
        """Return the next available index. If there are no more purge the genetic codes that have not been
        used in the longest time. DO NOT USE outside of the gene_pool_cache or genetic_code classes."""
        try:
            idx: int = super().next_index()
        except OverflowError:
            self.purge()
            idx = super().next_index()
        return idx

    def add(self, ggc: dict[str, Any]) -> int:
        """Add a dict type genetic code to the store."""
        return genetic_code(ggc).idx

    def update(self, ggcs: Iterable[dict[str, Any]]) -> list[int]:
        """Add a dict type genetic code to the store."""
        size_before: int = len(self)
        retval: list[int] = [genetic_code(o).idx for o in cast(Iterable[dict[str, Any]], ggcs)]
        size_after: int = len(self)
        _logger.info(f"Added {size_after - size_before} genetic codes to the GPC")
        return retval

    def reset(self, size: int | None = None) -> None:
        """A full reset of the store allows the size to be changed. All genetic codes
        are deleted which pushes the genetic codes to the genomic library as required.
        """
        super().reset(size)
        # Static store members
        self.ancestor_a: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.ancestor_b: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.e_count: NDArray[int32] = ones(self._size, dtype=int32)
        self.evolvability: NDArray[float32] = ones(self._size, dtype=float32)
        self.f_count: NDArray[int32] = ones(self._size, dtype=int32)
        self.fitness: NDArray[float32] = zeros(self._size, dtype=float32)
        self.gca: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.gcb: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.graph: NDArray[Any] = empty(self._size, dtype=graph)
        self.pgc: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)
        self.properties: NDArray[int64] = zeros(self._size, dtype=int64)
        self.reference_count: NDArray[int64] = zeros(self._size, dtype=int64)
        self.survivability: NDArray[float32] = zeros(self._size, dtype=float32)

        # Utility static store members
        # Access sequence of genetic codes. Used to determine which ones were least recently used.
        self.access_sequence: NDArray[int64] = full(self._size, INT64_MAX, dtype=int64)
        # Common dynamic store indices. -1 means not in the common dynamic store.
        self.common_ds_idx: NDArray[int32] = full(self._size, int32(-1), dtype=int32)
        self.genetic_code: NDArray[Any] = full(self._size, EMPTY_GENETIC_CODE, dtype=_genetic_code)

        # Re-initialize the common dynamic store wrapper
        for index_wrapper in self._common_ds_members.values():
            index_wrapper.index_mapping = self.common_ds_idx
        common_ds_index_wrapper.genetic_codes = self.genetic_code

        # Clean up the heap
        _logger.info("GPC reset to {self._size} entries and cleared.")
        _logger.debug(f"{collect()} unreachable objects not collected after reset.")

        # Total = 2* 8 + 5 * 4 = 36 bytes + base class per element

    def purge(self, fraction: float = 0.25) -> None:
        """Purge the store of unused data."""
        # Simply marking the data as unused is insufficient because the purged
        # data may be referenced by other objects. The purge function ensures that
        # all references to the purged data in the store are removed.
        num_to_purge: int = int(self._size * fraction)
        _logger.info(f"Purging {int(100 * fraction)}% = ({num_to_purge} of {self._size}) of the store")
        purge_indices: set[int] = set(argsort(self.access_sequence)[:num_to_purge])
        if _LOG_DEEP_DEBUG:
            _logger.info(f"Purging indices: {purge_indices}")
            _logger.debug(f"Access sequence numbers {self.access_sequence}")
        # Convert GC's with purged dependents into leaf nodes
        gc: _genetic_code
        for gc in self.genetic_code:
            gc.purge(purge_indices)

        # Push dirty genetic codes to the GP
        dirty_indices: set[int] = {idx for idx, byte in enumerate(self.status_byte) if byte & 1}
        self._push_to_gp(tuple(self.genetic_code[idx] for idx in dirty_indices))

        # Delete the purged objects
        for idx in purge_indices:
            del self[idx]

        # Clean up the heap: Intentionally calleding collect() regardless of the debug level.
        _logger.debug(f"{collect()} unreachable objects not collected after purge.")

    def optimize(self) -> None:
        """Optimize the store by looking for commonalities between genetic codes.
            1. Check to see if Leaf GC's have any dependents in the GPC to reference.
            2. If all of a Leaf GC's dependents are in the GPC then the leaf data can be deleted.
            3. Duplicate interfaces can be deleted.
            4. Duplicate rows can be deleted.
            5. Duplicate connections can be deleted.
            6. Duplicate graphs can be deleted.
        Try to minimize memory overhead by doing one at a time.
        NOTE: Optimizing the GPC does not delete any genetic codes.
        """
        # Make a dictionary of signatures to indices in the GPC
        sig_to_idx: dict[memoryview, int] = {gc["signature"].tobytes(): idx for idx, gc in enumerate(self.genetic_code) if gc.valid()}

        # #1 & #2
        # For every leaf GC check to see if any of its dependents are in the GPC
        # If they are then populate the object reference field
        count: int = 0
        for leaf in self.leaves():
            indices = tuple(sig_to_idx.get(self.genetic_code[leaf][field].tobytes(), -1) for field in STORE_PROXY_SIGNATURE_MEMBERS)
            for field, idx in (x for x in zip(STORE_PROXY_SIGNATURE_MEMBERS, indices) if x[1] >= 0):
                _logger.debug(f"Leaf {leaf} has a dependent in the GPC at index {idx} for field {field}")
                self[leaf][field] = self.genetic_code[idx]
            if all(idx >= 0 for idx in indices):
                del self._common_ds[self.common_ds_idx[leaf]]
                self.common_ds_idx[leaf] = -1
                count += 1
        _logger.info(f"Found {count} leaf genetic codes that need not be leaves.")

        # #3
        # Remove duplicate interfaces
        # NOTE: The hash of an interface is not the same as the instance of an interface.
        count: int = 0
        iface_to_iface: dict[interface, interface] = {}
        for gc in self.values():
            _rows: rows = gc["graph"].rows
            for row, iface in enumerate(_rows):
                if iface not in iface_to_iface:
                    iface_to_iface[iface] = iface
                else:
                    _rows[row] = iface_to_iface[iface]
                    count += 1
        _logger.info(f"Removed {count} duplicate interfaces.")

        # #4
        # Remove duplicate rows
        # NOTE: The hash of an row is not the same as the instance of a row.
        count: int = 0
        rows_to_rows: dict[rows, rows] = {}
        for gc in self.values():
            _rows: rows = gc["graph"].rows
            if _rows not in rows_to_rows:
                rows_to_rows[_rows] = _rows
            else:
                gc["graph"].rows = rows_to_rows[_rows]
                count += 1
        _logger.info(f"Removed {count} duplicate sets of rows.")

        # #5
        # Remove duplicate connections
        # NOTE: The hash of connections is not the same as the instance of connections.
        count: int = 0
        cons_to_cons: dict[connections, connections] = {}
        for gc in self.values():
            _graph = gc["graph"]
            cons: connections = _graph.connections
            if cons not in cons_to_cons:
                cons_to_cons[cons] = cons
            else:
                _graph.connections = cons_to_cons[cons]
                count += 1
        _logger.info(f"Removed {count} duplicate graph connection definitions.")

        # #6
        # Remove duplicate graphs
        # This is more efficient than duplicating the interfaces and connections
        # NOTE: The hash of a graph is not the same as the instance of a graph.
        count: int = 0
        graph_to_graph: dict[graph, graph] = {}
        for gc in self.values():
            _graph: graph = gc["graph"]
            if _graph not in graph_to_graph:
                graph_to_graph[_graph] = _graph
            else:
                gc["graph"] = graph_to_graph[_graph]
                count += 1
        _logger.info(f"Removed {count} duplicate graphs.")
        collect()

    def leaves(self) -> Iterator[int]:
        """Return each index of the leaf genetic codes."""
        # TODO: See how much faster this would be as numpy array manipulation
        for idx, _ in filter(lambda x: x[1] != -1, enumerate(self.common_ds_idx)):
            yield idx

    def signatures(self) -> Iterator[memoryview]:
        """Return the signatures of the genetic codes."""
        for gc in self.values():
            yield gc["signature"].data

    def values(self) -> Iterator[_genetic_code]:
        """Return the genetic codes."""
        # TODO: See how much faster this would be as numpy array manipulation
        # e.g. for gc in self.genetic_code[self.genetic_code != EMPTY_GENETIC_CODE & self.genetic_code != PURGED_GENETIC_CODE]
        for gc in filter(lambda x: x.valid(), self.genetic_code):
            yield gc
