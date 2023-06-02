"""Genetic Material Store.

The GMS is a abstract base class for retrieving genetic codes.
It defines common constants and methods for all GMSs.

The GC ancestory graph: A graph of the GCs ancestory. The GC ancestory graph is a directed graph with the GCs as nodes.
The GC structure graph: A graph of the GCs structure. The GC structure graph is a directed graph with the GCs as nodes.

Each graph is a view of the same base graph with edges labelled as ancestors or GC's.
"""

from json import load
from logging import DEBUG, NullHandler, getLogger, Logger
from os.path import dirname, join
from typing import Any, TypeVar, Generic
from graph_tool import Graph, GraphView
from .egp_typing import GmsGraphViews, AncestryKeys
from egp_types.aGC import aGC


# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# GMS graph view constants
_STRUCTURE_GRAPH = 'Structure'
_ANCESTRY_GRAPH = 'Ancestry'


# The update string
# No longer used but still here just in case of future need.
_WEIGHTED_VARIABLE_UPDATE_RAW = ('vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                                 ', 0.0::REAL, 0::INTEGER)')
_WEIGHTED_FIXED_UPDATE_RAW = 'weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC})'
_FIXED_UPDATE_RAW = 'fixed_array_update({CSCC}, {PSCC}, {CSPC})'
_SCALAR_COUNT_UPDATE = '{CSCC} + {PSCC} - {CSPC}'
_WEIGHTED_SCALAR_UPDATE = '({CSCV} * {CSCC} + {PSCV} * {PSCC} - {CSPV} * {CSPC}) / ' + _SCALAR_COUNT_UPDATE
_PGC_EVO_UPDATE_MAP: dict[str, str] = {
    'CSCV': 'EXCLUDED.{pgc_evolvability}',
    'CSCC': 'EXCLUDED.{pgc_e_count}',
    'PSCV': '"__table__".{pgc_evolvability}',
    'PSCC': '"__table__".{pgc_e_count}',
    'CSPV': 'EXCLUDED.{_pgc_evolvability}',
    'CSPC': 'EXCLUDED.{_pgc_e_count}'
}
_PGC_EVO_UPDATE_STR: str = '{pgc_evolvability} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_E_COUNT_UPDATE_STR: str = '{pgc_e_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_FIT_UPDATE_MAP: dict[str, str] = {
    'CSCV': 'EXCLUDED.{pgc_fitness}',
    'CSCC': 'EXCLUDED.{pgc_f_count}',
    'PSCV': '"__table__".{pgc_fitness}',
    'PSCC': '"__table__".{pgc_f_count}',
    'CSPV': 'EXCLUDED.{_pgc_fitness}',
    'CSPC': 'EXCLUDED.{_pgc_f_count}'
}
_PGC_FIT_UPDATE_STR: str = '{pgc_fitness} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_PGC_F_COUNT_UPDATE_STR: str = '{pgc_f_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_EVO_UPDATE_MAP: dict[str, str] = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{e_count}',
    'PSCV': '"__table__".{evolvability}',
    'PSCC': '"__table__".{e_count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_e_count}'
}
_EVO_UPDATE_STR: str = '{evolvability} = ' + _WEIGHTED_SCALAR_UPDATE.format_map(_EVO_UPDATE_MAP)
_EVO_COUNT_UPDATE_STR: str = '{e_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_EVO_UPDATE_MAP)
_REF_UPDATE_MAP: dict[str, str] = {
    'CSCC': 'EXCLUDED.{reference_count}',
    'PSCC': '"__table__".{reference_count}',
    'CSPC': 'EXCLUDED.{_reference_count}'
}
_REF_UPDATE_STR: str = '{reference_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_REF_UPDATE_MAP)
UPDATE_STR: str = ','.join((
    '{updated} = NOW()',
    _PGC_EVO_UPDATE_STR,
    _PGC_E_COUNT_UPDATE_STR,
    _PGC_FIT_UPDATE_STR,
    _PGC_F_COUNT_UPDATE_STR,
    _EVO_UPDATE_STR,
    _EVO_COUNT_UPDATE_STR,
    _REF_UPDATE_STR))


# Data schema
with open(join(dirname(__file__), "formats/gms_table_format.json"), "r", encoding="utf-8") as file_ptr:
    GMS_RAW_TABLE_SCHEMA: dict[str, Any] = load(file_ptr)


# The reference type
T = TypeVar('T')
U = TypeVar('U')


class genetic_material_store(Generic[T]):
    """Base class for all genetic material stores.

    The GMS provides graph based storage for genetic material. The graph is a directed acyclic graph (DAG) with
    multiple views (edge masks).
    NOTE: edges are all outbound from the GC node specified i.e. the edge to an ancestor is outbound from its decendant.
    """

    def __init__(self, keys: GmsGraphViews, node_key: str) -> None:
        """Initialise the GMS.

        Args
        ----
        edge_keys: The GC field keys used for the edges of the GC Structure and GC Ancestory views
        node_key: The GC field key used to record the GC node index
        """
        self._graph: Graph = Graph()
        self._node_key: str = node_key
        self._keys: GmsGraphViews = keys
        self._edge_ratio: int = sum(len(v) for v in keys.values())
        self._views: dict[str, GraphView] = {}

        # Set up properties for filtering edges in views
        self._graph.set_fast_edge_removal()
        for view in keys:
            self._graph.new_edge_property('bool', val=False)
            self._views[view] = GraphView(self._graph, efilt=self._graph.ep[view])

    def __getitem__(self, _: T) -> aGC:
        raise NotImplementedError

    def add(self, gcs: list | tuple[aGC, ...]) -> None:
        """Add GC nodes to the GMS graph

        Args
        ----
        gcs: GCs to add. Must have all self._edge_keys defined.
        """
        num: int = len(gcs)
        new_nodes = list(self._graph.add_vertex(num)) if num > 1 else [self._graph.add_vertex()]  # type: ignore

        # Add nodes first so that we can add edges
        for ngc, node in zip(gcs, new_nodes):
            ngc[self._node_key] = node

        # Add edges setting the edge property mask for the view
        for view, edges in self._keys.items():
            edge_view = self._graph.ep[view]
            for edge in edges['edges']:  # type: ignore edges is either StructureKeys or AncestoryKeys
                self._graph.add_edge_list([(gc[self._node_key], self[gc[edge]][self._node_key], True) for gc in gcs], eprops=(edge_view,))

    def remove(self, gcs: list | tuple[aGC, ...]) -> None:
        """Remove GC nodes from the GMS graph.

        A GC can only be removed if it has no incoming structural edges (i.e. it is not a sub-GC)
        and only has one descendant.

        Args
        ----
        gcs: GCs to remove.
        """
        
        for gc in gcs:


            self._graph.remove_vertex(gc[self._node_key])