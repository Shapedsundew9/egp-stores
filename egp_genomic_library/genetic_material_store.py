"""Genetic Material Store

The GMS is a abstract base class for retrieving genetic codes.
"""

from random import randint
from logging import DEBUG, NullHandler, getLogger
from networkx import MultiDiGraph, union, spring_layout, relabel_nodes, get_node_attributes


# Logging
_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)


# Constants
_GC_COUNT = 'gc_count'
_CODON_COUNT = 'num_codons'
_OBJECT = 'object'
_ZERO_GC_COUNT = {_GC_COUNT: 0, _CODON_COUNT: 0}


class genetic_material_store():

    def __init__(self, node_label='ref', left_edge_label='gca_ref', right_edge_label='gcb_ref'):
        """Define the keys used to build the GC graph."""
        self.nl = node_label
        self.lel = left_edge_label
        self.rel = right_edge_label
        self.graph = MultiDiGraph()

    def select(self):
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
        # Fast access
        nl = self.nl
        lel = self.lel
        rel = self.rel

        # Initialise search
        path = [node_label]
        idx = randint(0, self.graph.nodes[node_label][_OBJECT][_GC_COUNT] - 1)
        node = self.graph.nodes[node_label][_OBJECT]
        left_count = 0 if node[lel] is None else self.graph.nodes[node[lel]][_OBJECT][_GC_COUNT]
        pos = left_count

        while pos != idx:
            if pos > idx:
                path.append(node[lel])
                pos -= left_count
                node = self.graph.nodes[node[lel]][_OBJECT]
                left_count = 0 if node[lel] is None else self.graph.nodes[node[lel]][_OBJECT][_GC_COUNT]
                pos += left_count
            if pos < idx:
                path.append(node[rel])
                node = self.graph.nodes[node[rel]][_OBJECT]
                left_count = 0 if node[lel] is None else self.graph.nodes[node[lel]][_OBJECT][_GC_COUNT]
                pos += left_count + 1

        return path

    def add_nodes(self, gc_iter):
        """Add a GC nodes to the GMS graph.

        Adds all the GCs as nodes in gc_iter that do not already exist in the GMS graph.
        Edges are created between each GC added and its GCA & GCB sub-GC's if they are not None.
        Edges are directed from GC to sub-GC.

        gc_count is calculated for all added GC's.

        Args
        ----
        gc_iter (iter(gc)): Iterable of GC's to add. Must include self.nl, self.lel & self.rel keys.
        """
        # Fast access
        nl = self.nl
        lel = self.lel
        rel = self.rel

        # Add all node that do not already exist & the edges between them & existing nodes
        gc_list = [gc for gc in gc_iter if gc[nl] not in self.graph.nodes]
        self.graph.add_nodes_from(((gc[nl], {_OBJECT: gc}) for gc in gc_list))
        self.graph.add_edges_from(((gc[nl], gc[lel]) for gc in gc_list if gc[lel] is not None))
        self.graph.add_edges_from(((gc[nl], gc[rel]) for gc in gc_list if gc[rel] is not None))

        # Calculate node GC count
        # GC count is the number of nodes in the sub-tree including the root node.
        for gc in gc_list:
            work_stack = [gc]
            while work_stack:
                tgc = work_stack[-1]
                tgc_lel = tgc[lel]
                tgc_rel = tgc[rel]
                left_node = self.graph.nodes[tgc[lel]][_OBJECT] if tgc_lel is not None else _ZERO_GC_COUNT
                right_node = self.graph.nodes[tgc[rel]][_OBJECT] if tgc_rel is not None else _ZERO_GC_COUNT
                if _GC_COUNT not in left_node:
                    work_stack.append(left_node)
                elif _GC_COUNT not in right_node:
                    work_stack.append(right_node)
                else:
                    work_stack.pop()
                    tgc[_GC_COUNT] = left_node[_GC_COUNT] + right_node[_GC_COUNT] + 1
                    tgc[_CODON_COUNT] = 1 if tgc[_GC_COUNT] == 1 else left_node[_CODON_COUNT] + right_node[_CODON_COUNT]
