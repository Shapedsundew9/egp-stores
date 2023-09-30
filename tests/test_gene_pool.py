"""Test the Gene Pool."""
from logging import Logger, NullHandler, getLogger

from pypgtable.pypgtable_typing import TableConfigNorm

from egp_stores.egp_typing import GenePoolConfigNorm
from egp_stores.gene_pool import default_config as gp_default_config
from egp_stores.gene_pool import gene_pool
from egp_stores.genomic_library import default_config as gl_default_config
from egp_stores.genomic_library import genomic_library

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())



def test_default_instanciation() -> None:
    """Simple instanciation."""
    # Define genomic library configuration & instanciate
    gl_config: TableConfigNorm = gl_default_config()
    gl_config['delete_db'] = True
    glib: genomic_library = genomic_library(gl_config)

    # Establish the Gene Pool
    gp_config: GenePoolConfigNorm = gp_default_config()
    gpool: gene_pool = gene_pool({}, glib, gp_config)
    gpool.sub_process_init()

