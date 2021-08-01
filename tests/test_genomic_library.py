"""Test the genomic library."""

from random import choice
from genomic_library import genomic_library
from genomic_library import default_config
from genomic_library import sha256_to_str


def test_instanciation():
    """Create a genomic library."""
    config = default_config()
    config['database']['dbname'] = 'test_db'
    config['delete_table'] = True
    gl = genomic_library(config)
    assert gl


def test_select_getitem_encode():
    """Select the whole library then get a single item."""
    config = default_config()
    config['database']['dbname'] = 'test_db'
    config['delete_table'] = False
    gl = genomic_library(config)
    library = gl.select()
    gc = gl[choice(tuple(library.keys()))]
    assert gc['generation'] == 0
    assert bytearray(gc['signature']) == gl.encode_value('signature', sha256_to_str(gc['signature']))