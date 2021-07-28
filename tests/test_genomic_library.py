"""Test the genomic library."""

from genomic_library import genomic_library
from genomic_library import default_config


def test_instanciation():
    """Create a genomic library."""
    config = default_config()
    config['dbname'] = 'test_db'
    config['delete_table'] = True
    gl = genomic_library(config)
    assert gl