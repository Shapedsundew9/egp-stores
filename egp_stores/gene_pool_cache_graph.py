"""Gene Pool Cache Graph (GPCG) store.

The GPC maintains a graph of the relationships between GC's There are two types of graph:

The GC ancestory graph
The GC structure graph

Each graph is a view of the same base graph with edges labelled as ancestors or structure.
"""

from logging import DEBUG, Logger, NullHandler, getLogger
from typing import Any, Callable

from egp_types.aGC import aGC
from graph_tool import Graph, GraphView

# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# Key definitions
_REF = "ref"
_VIDX = "vertex_idx"
_ANCESTOR_A = "ancestor_a_ref"
_ANCESTOR_B = "ancestor_b_ref"
_MISSING_A = "missing_links_a"
_MISSING_B = "missing_links_b"
_LOST_D = "lost_descendants"
_GCA = "gca_ref"
_GCB = "gcb_ref"
_STRUCTURE = "structure"
_ANCESTRY = "ancestry"
_R_DIST_A = "r_dist_a"
_R_DIST_B = "r_dist_b"


# Default
def _from_higher_layer(_: int) -> bool:
    """Default function to test if a GC is from a higher layer."""
    raise NotImplementedError


class gene_pool_cache_graph:
    """Provides graph based storage for genetic material.

    NOTE: edges are all outbound from the GC node specified i.e. the edge to an ancestor is outbound from its decendant.
    """

    def __init__(self) -> None:
        """Initialise the graph."""
        self._graph: Graph = Graph(directed=False)
        self._graph.vp["ref"] = self._graph.new_vertex_property("int64_t", val=0)
        self._vacant_indices: list[int] = []
        self.refs: Any = self._graph.vp["ref"]

        # Defined by the GP sub-process initialisation function
        self.from_higher_layer: Callable[[int], bool] = _from_higher_layer

        # Ancestry & structure views of the graph
        self._graph.ep[_STRUCTURE] = self._graph.new_edge_property("bool", val=False)
        self._structure = GraphView(self._graph, efilt=self._graph.ep[_STRUCTURE])
        self._graph.ep[_ANCESTRY] = self._graph.new_edge_property("bool", val=False)
        self._ancestry = GraphView(self._graph, efilt=self._graph.ep[_ANCESTRY])

        # Edge weights for relationship distances measurements
        # Direct decesendants have a distance of 1
        self._graph.ep[_R_DIST_A] = self._graph.new_edge_property("int32_t", val=1)
        self._graph.ep[_R_DIST_B] = self._graph.new_edge_property("int32_t", val=1)

    def __getitem__(self, _: int) -> aGC:
        """Get a GC from the GPC.

        Args
        ----
        _: Reference of GC to get.
        """
        raise NotImplementedError

    def add(self, gcs: list[aGC]) -> None:
        """Add GC nodes to the GMS graph

        Args
        ----
        gcs: GCs to add. Must have all self._edge_keys defined.
        """

        # Fill vacant indices first
        new_indices: list[int] = []
        while gcs and self._vacant_indices:
            new_idx: int = self._vacant_indices.pop()
            ngc: aGC = gcs.pop()
            ngc[_REF] = new_idx
            new_indices.append(new_idx)
            self._graph.vp["ref"][new_idx] = ngc[_REF]

        # Create new indices if necessary
        if gcs:
            num: int = len(gcs)
            new_vertices = list(self._graph.add_vertex(num)) if num > 1 else [self._graph.add_vertex()]  # type: ignore
            for vertex, ngc in zip(new_vertices, gcs):
                self._graph.vp["ref"][vertex] = ngc[_REF]
            new_indices.extend(map(int, new_vertices))  # type: ignore

        # Associate the new indices with the GCs
        for ngc, idx in zip(gcs, new_indices):
            ngc[_VIDX] = idx

        # Add edges setting the edge property mask for the view
        ancestry_edge_list = [
            (ngc.get(_VIDX, -1), self[ngc[_ANCESTOR_A]].get(_VIDX, -1), True)
            for ngc in gcs
            if ngc[_ANCESTOR_A]
        ]
        ancestry_edge_list.extend(
            [
                (ngc.get(_VIDX, -1), self[ngc[_ANCESTOR_B]].get(_VIDX, -1), True)
                for ngc in gcs
                if ngc[_ANCESTOR_B]
            ]
        )
        self._graph.add_edge_list(
            ancestry_edge_list, eprops=(self._graph.ep[_ANCESTRY],)
        )
        structure_edge_list = [
            (ngc.get(_VIDX, -1), self[ngc[_GCA]].get(_VIDX, -1), True)
            for ngc in gcs
            if ngc[_GCA]
        ]
        structure_edge_list.extend(
            [
                (ngc.get(_VIDX, -1), self[ngc[_GCB]].get(_VIDX, -1), True)
                for ngc in gcs
                if ngc[_GCB]
            ]
        )
        self._graph.add_edge_list(
            structure_edge_list, eprops=(self._graph.ep[_STRUCTURE],)
        )

    def remove(self, refs: list[int]) -> list[int]:
        """Remove GC nodes from the GMS graph.

        Nodes can only be removed from the GMS graph if they have no incoming structure edges and
        are not from a higher layer.
        Note that nodes are not deleted just the edges removed and the node marked as vacant..get(_MISSING_B, 0)

        Args
        ----
        gcs: References of GCs to remove.
        """
        in_struct_edges = self._graph.get_in_degrees(
            [self[ref].get(_VIDX, -1) for ref in refs], self._graph.ep["Structure"]
        )
        fhl_list = [self.from_higher_layer(ref) for ref in refs]
        victims: list[aGC] = [
            self[ref]
            for ref, ise, fhl in zip(refs, in_struct_edges, fhl_list)
            if not ise and not fhl
        ]
        victims_refs = []
        while victims:
            vgc: aGC = victims.pop()
            vgc_idx = vgc.get(_VIDX, -1)
            vgc_ref = vgc[_REF]
            ancestor_a_ref = vgc[_ANCESTOR_A]

            # If this GC has no ancestor A then it is a codon and cannot be removed
            if not ancestor_a_ref:
                continue

            # Otherwise it is fair game
            victims_refs.append(vgc_ref)
            ancestor_a_idx = self[vgc[_ANCESTOR_A]].get(_VIDX, -1)
            ancestor_a = self[ancestor_a_ref]
            ancestor_b_ref = vgc[_ANCESTOR_B]

            # Remove ancestry edges & update missing & lost GC's
            # TODO: Tests
            #   1. Total missing + lost + existing in graph = total GCs ever added
            #   2. Relationship distances meausred by graph tool in the full graph should match recorded distances in trimmed graph
            #   3. Relationship distances measured by graph tool in the full graph should match those measured in the trimmed graph
            descendant_edges = [
                e
                for e in self._graph.vertex(vgc_idx).in_edges()
                if self._graph.ep["ancestry"][e]
            ]
            ancestor_a[_LOST_D] = (
                ancestor_a.get(_LOST_D, 0)
                + vgc.get(_MISSING_A, 0)
                + vgc.get(_LOST_D, 0)
                + 1
            )
            for descendant_edge in descendant_edges:
                descendant_idx = descendant_edge.source()
                descendant_ref = self.refs[descendant_idx]
                descendant = self[descendant_ref]
                self._graph.remove_edge(descendant_edge)
                new_edge = self._graph.add_edge(descendant_idx, ancestor_a_idx)
                self._graph.ep[_ANCESTRY][new_edge] = True
                if descendant[_ANCESTOR_A] == vgc_ref:
                    descendant[_ANCESTOR_A] = ancestor_a_ref
                    descendant[_MISSING_A] = (
                        descendant.get(_MISSING_A, 0) + vgc.get(_MISSING_A, 0) + 1
                    )
                    self._graph.ep[_R_DIST_A][new_edge] = (
                        descendant.get(_MISSING_A, 0) + 1
                    )
                else:
                    descendant[_ANCESTOR_B] = ancestor_a_ref
                    descendant[_MISSING_B] = (
                        descendant.get(_MISSING_B, 0) + vgc.get(_MISSING_A, 0) + 1
                    )
                    self._graph.ep[_R_DIST_B][new_edge] = (
                        descendant.get(_MISSING_A, 0) + 1
                    )

            # If there is an ancestor B then it may have lost descendants (those that were missing links)
            if ancestor_b_ref:
                ancestor_b = self[ancestor_b_ref]
                ancestor_b[_LOST_D] = ancestor_b.get(_LOST_D, 0) + vgc.get(
                    _MISSING_B, 0
                )
                assert (
                    vgc.get(_MISSING_B, 0) == 0
                ), "GC has no ancestor B but has missing links B!"

            # Remove structure edges
            sep = self._graph.ep[_STRUCTURE]
            for structure_edge in [
                e
                for e in self._graph.vertex(vgc_idx).out_edges()
                if self._graph.ep[_STRUCTURE][e]
            ]:
                structure_idx = structure_edge.target()
                vertex = self._graph.vertex(structure_idx)
                sgc_ref = self.refs[structure_idx]
                sgc = self[sgc_ref]

                # If there is more than 1 incoming structure edge the sGC is not removed.
                # GC's imported from higher layers are not removed even if they have only 1 incoming structure edge.
                if sum(
                    sep[e] for e in vertex.in_edges()
                ) == 1 and not self.from_higher_layer(sgc_ref):
                    victims.append(sgc)

                # Remove the structure edge
                self._graph.remove_edge(structure_edge)

            # Sanity checks
            if _LOG_DEBUG:
                assert not self.from_higher_layer(
                    vgc_ref
                ), "GC being removed is from a higher layer!"
                assert (
                    self._graph.vertex(vgc_idx).in_degree() == 0
                ), "GC being removed has incoming edges!"
                assert (
                    self._graph.vertex(vgc_idx).out_degree() == 0
                ), "GC being removed has outgoing edges!"

            # Mark the vertex as vacant
            vgc[_VIDX] = -1
            self._vacant_indices.append(vgc_idx)

        # The actual victims may not include all the GCs that were attempted to be removed
        # and may include GCs that were not removed due to having no incoming structure edges after other removals
        return victims_refs
