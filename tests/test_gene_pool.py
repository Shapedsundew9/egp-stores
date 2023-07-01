"""Test the Gene Pool."""
from logging import Logger, NullHandler, getLogger
from os.path import dirname, join
from typing import Any
from egp_population.population import (configure_populations,
                                       population_table_config)
from egp_population.egp_typing import PopulationConfigNorm
from egp_stores.gene_pool import default_config as gp_default_config
from egp_stores.gene_pool import gene_pool
from egp_stores.genomic_library import default_config as gl_default_config
from egp_stores.genomic_library import genomic_library
from egp_stores.egp_typing import GenePoolConfigNorm
from pypgtable import table
from pypgtable.typing import TableConfigNorm


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


def test_default_instanciation() -> None:
    """Simple instanciation."""

    # Define gene pool configuration
    gp_config: GenePoolConfigNorm = gp_default_config()

    # Define population configuration
    p_table_config: TableConfigNorm = population_table_config()

    # Define genomic library configuration & instanciate
    gl_config: dict[str, Any] = gl_default_config()
    gl_config['database']['dbname'] = 'simple_db'
    gl_config['delete_table'] = True
    glib: genomic_library = genomic_library(gl_config, [join(dirname(__file__), 'data/simple_gl.json')])

    # Get the population configurations
    p_config_tuple: tuple[dict[int, PopulationConfigNorm], table, table] = configure_populations(config['population'], p_table_config)
    p_configs: dict[int, PopulationConfigNorm] = p_config_tuple[0]
    p_table: table = p_config_tuple[1]
    pm_table: table = p_config_tuple[2]

    # Establish the Gene Pool
    gpool: gene_pool = gene_pool(p_configs, glib, gp_config)