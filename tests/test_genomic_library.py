"""Test the genomic library."""

from random import choice

from egp_stores import default_config, genomic_library, sha256_to_str


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
    gc = gl[choice(tuple(library))['signature']]
    assert gc['generation'] == 0
    assert bytearray(gc['signature']) == gl.encode_value('signature', sha256_to_str(gc['signature']))
