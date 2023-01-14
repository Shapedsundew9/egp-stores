"""Test the genomic library."""

from random import choice
from typing import Any

from egp_stores import default_config, genomic_library
from egp_types.conversions import sha256_to_str
from pypgtable.typing import RowIter


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
