"""Genetic Material Store.

The GMS is a abstract base class for retrieving genetic codes.
"""

from random import randint
from logging import DEBUG, NullHandler, getLogger
from graph_tool import Graph, Vertex
from graph_tool.topology import label_out_component, all_paths
from numpy.random import choice as weighted_choice
from random import choice
from json import load
from os.path import dirname, join


# Logging
_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)


# Constants
_GC_DEPTH = 'code_depth'
_CODON_DEPTH = 'codon_depth'
_GC_COUNT = 'num_codes'
_CODON_COUNT = 'num_codons'
_UNIQUE_GC_COUNT = 'num_unique_codes'
_UNIQUE_CODON_COUNT = 'num_unique_codons'
_OBJECT = 'object'
_ZERO_GC_COUNT = {_GC_COUNT: 0, _CODON_COUNT: 0, _GC_DEPTH: 0, _CODON_DEPTH: 0, _UNIQUE_GC_COUNT: 0, _UNIQUE_CODON_COUNT: 0}

# The update string
# No longer used but still here just in case of future need.
_WEIGHTED_VARIABLE_UPDATE_RAW = ('vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                   ', 0.0::REAL, 0::INTEGER)')
_WEIGHTED_FIXED_UPDATE_RAW = ('weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC})')
_FIXED_UPDATE_RAW = 'fixed_array_update({CSCC}, {PSCC}, {CSPC})'
_SCALAR_COUNT_UPDATE = '{CSCC} + {PSCC} - {CSPC}'
_WEIGHTED_SCALAR_UPDATE = '({CSCV} * {CSCC} + {PSCV} * {PSCC} - {CSPV} * {CSPC}) / ' + _SCALAR_COUNT_UPDATE
_PGC_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{pgc_evolvability}',
    'CSCC': 'EXCLUDED.{pgc_e_count}',
    'PSCV': '"__table__".{pgc_evolvability}',
    'PSCC': '"__table__".{pgc_e_count}',
    'CSPV': 'EXCLUDED.{_pgc_evolvability}',
    'CSPC': 'EXCLUDED.{_pgc_e_count}'
}
_PGC_EVO_UPDATE_STR = '{pgc_evolvability} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_E_COUNT_UPDATE_STR = '{pgc_e_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_FIT_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{pgc_fitness}',
    'CSCC': 'EXCLUDED.{pgc_f_count}',
    'PSCV': '"__table__".{pgc_fitness}',
    'PSCC': '"__table__".{pgc_f_count}',
    'CSPV': 'EXCLUDED.{_pgc_fitness}',
    'CSPC': 'EXCLUDED.{_pgc_f_count}'
}
_PGC_FIT_UPDATE_STR = '{pgc_fitness} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_PGC_F_COUNT_UPDATE_STR = '{pgc_f_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{e_count}',
    'PSCV': '"__table__".{evolvability}',
    'PSCC': '"__table__".{e_count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_e_count}'
}
_EVO_UPDATE_STR = '{evolvability} = ' + _WEIGHTED_SCALAR_UPDATE.format_map(_EVO_UPDATE_MAP)
_EVO_COUNT_UPDATE_STR = '{e_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_EVO_UPDATE_MAP)
_REF_UPDATE_MAP = {
    'CSCC': 'EXCLUDED.{reference_count}',
    'PSCC': '"__table__".{reference_count}',
    'CSPC': 'EXCLUDED.{_reference_count}'
}
_REF_UPDATE_STR = '{reference_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_REF_UPDATE_MAP)
UPDATE_STR = ','.join((
    '{updated} = NOW()',
    _PGC_EVO_UPDATE_STR,
    _PGC_E_COUNT_UPDATE_STR,
    _PGC_FIT_UPDATE_STR,
    _PGC_F_COUNT_UPDATE_STR,
    _EVO_UPDATE_STR,
    _EVO_COUNT_UPDATE_STR,
    _REF_UPDATE_STR))


# Data schema
with open(join(dirname(__file__), "formats/gms_table_format.json"), "r") as file_ptr:
    GMS_TABLE_SCHEMA = load(file_ptr)


class genetic_material_store():
    """Base class for all genetic material stores."""

    def __init__(self, node_label='ref', left_edge_label='gca_ref', right_edge_label='gcb_ref'):
        """Define the keys used to build the GC graph."""
        self.nl = node_label
        self.lel = left_edge_label
        self.rel = right_edge_label
        self.graph = Graph()
        self.graph.vertex_properties[_OBJECT] = self.graph.new_vertex_property('python::object')
        self._l2v = {}

    def select(self):
        """Select method required for all GMSs."""
        raise NotImplementedError

    def random_descendant(self, node_label):
        """Select a random descendant from the node with node_label.

        nl may be selected.
        Every node in the tree has a weight equal to the number of incoming edges. i.e. if a node
        appears N times in the flattened tree it will have a weight of N.

        Args
        ----
        nl (type(gc[nl])): The label of the node at the root of the tree to select from.

        Returns
        -------
        list(type(gc[nl])): The path to the selected node. The first element in the list will be nl.
        """
        source_node = self._l2v[node_label]
        subtree_mask = label_out_component(self.graph, source_node).a
        random_node_idx = weighted_choice(len(subtree_mask), p = subtree_mask / subtree_mask.sum())
        all_paths_tuple = tuple(all_paths(self.graph, source_node, random_node_idx))
        if not all_paths_tuple:
            return [node_label]
        path_indices = choice(all_paths_tuple)
        return [self.graph.vertex_properties[_OBJECT][v][self.nl] for v in path_indices]

    def remove_nodes(self, nl_iter):
        """Remove nodes recursively from the graph.

        If a node in the gc_iter has no incoming connections then it is removed from the graph along
        with all edges eminating from it. If that removes all incoming edges from another node then
        that node is also removed.

        Args
        ----
        nl_iter(iter(nl)): The list of GC node labels to remove if not the destination of an edge.

        Returns
        -------
        list(gc[nl]): The list of GC node labels actually deleted.
        """
        # All the nodes that exist in the graph and have no incoming edges
        v_list = [self._l2v[nl] for nl in nl_iter if nl in self._l2v and self._l2v[nl].in_degree()]

        # Use a while loop so we can add to it
        victims = set()
        while v_list:
            v = v_list.pop(0)
            victims.add(v)
            for e in v.out_edges():
                if e.target().in_degree() == 1:
                    v_list.append(e.target())

        # Remove all the victims
        self.graph.remove_vertex(victims, fast=True)

        # Return list of victim node labels
        object_property = self.graph.vertex_properties[_OBJECT]
        return [object_property[v][self.nl] for v in victims]

    def add_nodes(self, gc_iter):
        """Add a GC nodes to the GMS graph.

        Adds all the GCs as nodes in gc_iter that do not already exist in the GMS graph.
        Edges are created between each GC added and its GCA & GCB sub-GC's if they are not None.
        Edges are directed from GC to sub-GC.

        NOTE: Assumes all referenced GCs in gc_iter exist in self.graph or are in gc_iter.

        _GC_COUNT, _CODON_COUNT, _GC_DEPTH & _CODON_DEPTH are calculated for all added GC's.

        Args
        ----
        gc_iter (iter(gc)): Iterable of GC's to add. Must include self.nl, self.lel & self.rel keys.
        """
        # Fast access
        nl = self.nl
        lel = self.lel
        rel = self.rel

        # Add all nodes that do not already exist
        gc_list = [gc for gc in gc_iter if gc[nl] not in self._l2v]
        new_vertices = self.graph.add_vertex(len(gc_list))

        # Special case when gc_list only has one element - annoying graph_tool function behaviour
        if isinstance(new_vertices, Vertex):
            new_vertices = [new_vertices]
            
        for nv, gc in zip(new_vertices, gc_list):
            self.graph.vertex_properties[_OBJECT][nv] = gc
            self._l2v[gc[nl]] = nv # Needs to be the descriptor as fast removal invalidate indices.

        self.graph.add_edge_list(((self._l2v[gc[nl]], self._l2v[gc[lel]]) for gc in gc_list if gc[lel] is not None))
        self.graph.add_edge_list(((self._l2v[gc[nl]], self._l2v[gc[rel]]) for gc in gc_list if gc[rel] is not None))

        # Calculate node GC count
        # _GC_COUNT is the number of nodes in the sub-tree including the root node.
        # _CODON_COUNT is the number of codons in the sub-tree including the root node.
        # _GC_DEPTH is the depth of the GC tree in codes.
        # _CODON_DEPTH is the depth of the codon tree.
        for gc in gc_list:
            # FIXME: This is a placeholder. Figuring out the codon depth requires building the codon tree
            # Is that worth it (memory & cpu)?
            gc[_CODON_DEPTH] = 1
            # FIXME: These are placeholders too. Figuring out the unique number of codes and codons requires
            # an expensive operation each time.  Is that worth it (memory & cpu)?
            gc[_UNIQUE_GC_COUNT] = 1
            gc[_UNIQUE_CODON_COUNT] = 1

            work_stack = [gc]
            if _LOG_DEBUG:
                _logger.debug(f'Adding node {gc[nl]}. Work stack depth 1.')
            while work_stack:
                tgc = work_stack[-1]
                tgc_lel = tgc[lel]
                tgc_rel = tgc[rel]
                left_node = self.graph.vertex_properties[_OBJECT][self._l2v[tgc_lel]] if tgc_lel is not None else _ZERO_GC_COUNT
                right_node = self.graph.vertex_properties[_OBJECT][self._l2v[tgc_rel]] if tgc_rel is not None else _ZERO_GC_COUNT
                if _GC_COUNT not in left_node:
                    work_stack.append(left_node)
                    if _LOG_DEBUG:
                        _logger.debug(f'Adding node left {left_node[nl]}, work_stack length {len(work_stack)}.')
                elif _GC_COUNT not in right_node:
                    work_stack.append(right_node)
                    if _LOG_DEBUG:
                        _logger.debug(f'Adding node right {right_node[nl]}, work_stack length {len(work_stack)}.')
                else:
                    work_stack.pop()
                    tgc[_GC_COUNT] = left_node[_GC_COUNT] + right_node[_GC_COUNT] + 1
                    tgc[_GC_DEPTH] = left_node[_GC_DEPTH] if right_node[_GC_DEPTH] < left_node[_GC_DEPTH] else right_node[_GC_DEPTH]
                    tgc[_GC_DEPTH] += 1
                    tgc[_CODON_COUNT] = 1 if tgc[_GC_COUNT] == 1 else left_node[_CODON_COUNT] + right_node[_CODON_COUNT]
                    if _LOG_DEBUG:
                        _logger.debug(f'Leaf node popped {tgc[nl]}, work_stack length {len(work_stack)}, count {tgc[_GC_COUNT]}.')

    def hl_copy(self, gcs, field_names):
        """Copy the higher layer field to the current layer field.

        gc is modified.
        Current layer fields will be created if they do not exist.

        A higher layer field starts with an underscore '_' and has an underscoreless counterpart.
        e.g. '_field' and 'field'. The _field holds the value of field when the GC was pulled from
        the higher layer. i.e. after being pulled from the GMS _field must = field. field can then
        be modified by the lower layer. NB: Updating the value back into the GMS is a bit more complex.

        FYI: This is not an automatic step as it deletes information i.e. the lower layer may care what
        the higher layers higher layer values are.

        Args
        ----
        gcs (iter(dict)): Dictionary containing field_names fields (typically a GC)
        field_names (iter(str)): List of valid higher layer field names. i.e. start with an underscore.
        """
        for gc in gcs:
            gc.update({k: gc[k[1:]] for k in field_names})

"""
Some benchmarking on SHA256 generation
======================================
Python 3.8.5

>>> def a():
...     start = time()
...     for _ in range(10000000): int(sha256("".join(string.split()).encode()).hexdigest(), 16)
...     print(time() - start)
...
>>> a()
8.618626356124878
>>> def b():
...     start = time()
...     for _ in range(10000000): int.from_bytes(sha256("".join(string.split()).encode()).digest(), 'big')
...     print(time() - start)
...
>>> b()
7.211490631103516
>>> def c():
...     start = time()
...     for _ in range(10000000): sha256("".join(string.split()).encode()).hexdigest()
...     print(time() - start)
...
>>> c()
6.463267803192139
>>> def d():
...     start = time()
...     for _ in range(10000000): sha256("".join(string.split()).encode()).digest()
...     print(time() - start)
...
>>> d()
6.043259143829346
>>> def e():
...     start = time()
...     for _ in range(10000000): {sha256("".join(string.split()).encode()).digest(): "Test"}
...     print(time() - start)
...
>>> e()
6.640311002731323
>>> def f():
...     start = time()
...     for _ in range(10000000): {int.from_bytes(sha256("".join(string.split()).encode()).digest(), 'big'): "Test"}
...     print(time() - start)
...
>>> f()
7.6320412158966064
>>> def g():
...     start = time()
...     for _ in range(10000000): {sha256("".join(string.split()).encode()).hexdigest(): "Test"}
...     print(time() - start)
...
>>> g()
7.144319295883179
>>> def h1():
...     start = time()
...     for _ in range(10000000): getrandbits(256)
...     print(time() - start)
...
>>> h1()
1.0232288837432861
>>> def h2():
...     start = time()
...     for _ in range(10000000): getrandbits(128)
...     print(time() - start)
...
>>> h2()
0.8551476001739502
>>> def h3():
...     start = time()
...     for _ in range(10000000): getrandbits(64)
...     print(time() - start)
...
>>> h3()
0.764052152633667
>>> def i():
...     start = time()
...     for _ in range(10000000): getrandbits(256).to_bytes(32, 'big')
...     print(time() - start)
...
>>> i()
2.038336753845215
"""


"""
Some Benchmarking on hashing SHA256
===================================
Python 3.8.5

>>> a =tuple( (getrandbits(256).to_bytes(32, 'big') for _ in range(10000000)))
>>> b =tuple( (int(getrandbits(63)) for _ in range(10000000)))
>>> start = time(); c=set(a); print(time() - start)
1.8097834587097168
>>> start = time(); d=set(b); print(time() - start)
1.0908379554748535
"""
