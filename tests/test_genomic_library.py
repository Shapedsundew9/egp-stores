"""Test the genomic library."""

from random import choice
from typing import Any
from os.path import dirname, join

from egp_types.conversions import sha256_to_str
from pypgtable.typing import RowIter

from egp_stores import default_config, genomic_library


def test_instanciation() -> None:
    """Create a genomic library."""
    config: dict[str, Any] = default_config()
    config['database']['dbname'] = 'test_db'
    config['delete_table'] = True
    gl: genomic_library = genomic_library(config)
    assert gl


def test_select_getitem_encode() -> None:
    """Select the whole library then get a single item."""
    config: dict[str, Any] = default_config()
    config['database']['dbname'] = 'test_db'
    config['delete_table'] = False
    gl: genomic_library = genomic_library(config)
    library: RowIter = gl.select()
    gc: dict[str, Any] = gl[choice(tuple(library))['signature']]
    assert gc['generation'] == 0
    assert bytearray(gc['signature']) == gl.encode_value('signature', sha256_to_str(gc['signature']))


def test_simple_GL_load() -> None:
    """Load the simple GL created by
    https://github.com/Shapedsundew9/experiments/blob/main/genomic-library-generators/simple_generator.py)"""
    config: dict[str, Any] = default_config()
    config['database']['dbname'] = 'simple_db'
    config['delete_table'] = True
    gl: genomic_library = genomic_library(config, [join(dirname(__file__), 'data/simple_gl.json')])
    assert gl
