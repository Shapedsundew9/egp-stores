"""Test the Gene Pool."""
from logging import Logger, NullHandler, getLogger
from os.path import dirname, join
from typing import Any
from uuid import uuid4

from egp_population.egp_typing import PopulationConfigNorm
from egp_population.population_config import (
    configure_populations,
    population_table_config,
)
from pypgtable import table
from pypgtable.pypgtable_typing import TableConfigNorm

from egp_stores.egp_typing import GenePoolConfigNorm
from egp_stores.gene_pool import default_config as gp_default_config
from egp_stores.gene_pool import gene_pool
from egp_stores.genomic_library import default_config as gl_default_config
from egp_stores.genomic_library import genomic_library

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


_PS_CONFIG = {
    "configs": [
        {
            "git-repo": "egp-playground",
            "git_url": "https://github.com/Shapedsundew9",
            "git_hash": "6d9ab8eddd0861a5339886df6edaaf2e11db2685",
            "worker_id": uuid4(),
            "size": 4,
            "inputs": ["int", "int"],
            "outputs": ["int"],
            "description": "Trial!",
            "preload_import": "number_tree.number_tree",
            "fitness_import": "number_tree.number_tree",
            "survivability_import": "number_tree.number_tree",
        }
    ]
}


def test_default_instanciation() -> None:
    """Simple instanciation."""

    # Define gene pool configuration
    gp_config: GenePoolConfigNorm = gp_default_config()

    # Define population configuration
    p_table_config: TableConfigNorm = population_table_config()

    # Define genomic library configuration & instanciate
    gl_config: dict[str, Any] = gl_default_config()
    gl_config["database"]["dbname"] = "simple_db"
    gl_config["delete_table"] = True
    glib: genomic_library = genomic_library(
        gl_config, [join(dirname(__file__), "data/simple_gl.json")]
    )

    # Get the population configurations
    p_config_tuple: tuple[
        dict[int, PopulationConfigNorm], table, table
    ] = configure_populations(_PS_CONFIG, p_table_config)
    p_configs: dict[int, PopulationConfigNorm] = p_config_tuple[0]
    p_table: table = p_config_tuple[1]
    pm_table: table = p_config_tuple[2]

    # Establish the Gene Pool
    gpool: gene_pool = gene_pool(p_configs, glib, gp_config)
