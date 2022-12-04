"""Test the Genetic Material Store (GMS)"""
import gi
gi.require_version('Gtk', '3.0')

import pytest
from random import choice
from os.path import dirname, join
from json import load
from logging import NullHandler, getLogger
from graph_tool.draw import graph_draw

from egp_stores import genetic_material_store, _CODON_COUNT, _GC_COUNT
from egp_stores.genetic_material_store import _OBJECT

_logger = getLogger(__name__)
_logger.addHandler(NullHandler())


# Load test data
with open(join(dirname(__file__), 'data/genetic_material_store_test_cases.json'), 'r') as fileptr:
    test_data = load(fileptr)


def test_instanciation():
    """Straight up intanciation test."""
    gms = genetic_material_store()
    assert gms


@pytest.mark.parametrize("test_case_idx", range(len(test_data)))
def test_graph_construction(test_case_idx):
    """Construct graphs and verify meta data is as expected."""
    gms = genetic_material_store("id", "left", "right")
    test_case = test_data[test_case_idx]
    gms.add_nodes(test_case)
    graph_draw(
        gms.graph,
        output_size=(1000, 1000),
        vertex_size=50,
        vertex_text=gms.graph.new_vertex_property("string", [gc[gms.nl] for gc in gms.graph.vertex_properties[_OBJECT]]),
        edge_pen_width=8,
        output=join(dirname(__file__), f'../logs/test_graph_construction_{test_case_idx}.png')
    )
    for node in test_case:
        if node['expected_gc_count'] != node[_GC_COUNT]:
            _logger.error(f'Unexpected gc_count in {node}')
        if node['expected_codons'] != node[_CODON_COUNT]:
            _logger.error(f'Unexpected gc_count in {node}')
        assert node['expected_gc_count'] == node[_GC_COUNT]
        assert node['expected_codons'] == node[_CODON_COUNT]


def test_root_random_descendant():
    """Select a random descendant from root node.

    Validation is done by verifying the all paths between nodes exist.
    1000 iterations gives a 1 in 10**23 chance of a node in a 20 node graph not being chosen.
    """
    gms = genetic_material_store("id", "left", "right")
    gms.add_nodes(test_data[2])
    all_paths = {'A', 'AB', 'ABC', 'ABCD', 'ABCDE', 'ABCG', 'ABCGF', 'ABCGH', 'ABI', 'AJ',
        'AJK', 'AJKL', 'AJKLM', 'AJKLN', 'AJKP', 'AJKPQ', 'AJKPQR', 'AJKPQRS', 'AJKPT', 'AJO'}
    paths = {''.join(gms.random_descendant(test_data[2][0]['id'])) for _ in range(1000)}
    assert all_paths == paths


def test_random_descendant():
    """Select a random descendant.

    Validation is done by verifying all paths between all nodes exist.
    4000 iterations gives a 1 in 10**22 chance of a 1 path in 81 not being chosen
    """
    gms = genetic_material_store("id", "left", "right")
    gms.add_nodes(test_data[2])
    all_paths = {'A', 'AB', 'ABC', 'ABCD', 'ABCDE', 'ABCG', 'ABCGF', 'ABCGH', 'ABI', 'AJ', 'AJK',
        'AJKL', 'AJKLM', 'AJKLN', 'AJKP', 'AJKPQ', 'AJKPQR', 'AJKPQRS', 'AJKPT', 'AJO', 'B', 'BC',
        'BCD', 'BCDE', 'BCG', 'BCGF', 'BCGH', 'BI', 'C', 'CD', 'CDE', 'CG', 'CGF', 'CGH', 'D',
        'DE', 'E', 'F', 'G', 'GF', 'GH', 'H', 'I', 'J', 'JK', 'JKL', 'JKLM', 'JKLN', 'JKP', 'JKPQ',
        'JKPQR', 'JKPQRS', 'JKPT', 'JO', 'K', 'KL', 'KLM', 'KLN', 'KP', 'KPQ', 'KPQR', 'KPQRS',
        'KPT', 'L', 'LM', 'LN', 'M', 'N', 'O', 'P', 'PQ', 'PQR', 'PQRS', 'PT', 'Q', 'QR', 'QRS',
        'R', 'RS', 'S', 'T'}
    paths = {''.join(gms.random_descendant(choice(test_data[2])['id'])) for _ in range(4000)}
    assert all_paths == paths
