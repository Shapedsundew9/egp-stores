"""Test the Gene Pool."""
from logging import INFO, CRITICAL, Logger, NullHandler, getLogger, basicConfig
from typing import Any
from numpy import ndarray

from pypgtable.pypgtable_typing import TableConfigNorm

from egp_stores.egp_typing import GenePoolConfigNorm
from egp_stores.gene_pool import default_config as gp_default_config
from egp_stores.gene_pool import gene_pool
from egp_stores.genomic_library import default_config as gl_default_config
from egp_stores.genomic_library import genomic_library
from egp_stores.gene_pool_cache import STORE_ALL_MEMBERS, _genetic_code, EMPTY_GENETIC_CODE


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
getLogger("pypgtable.database").setLevel(INFO)
getLogger("obscure_password").setLevel(INFO)
if __name__ == "__main__":
    basicConfig(filename="test_gene_pool.log", filemode='w',
        format='%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s',datefmt='%H:%M:%S', level=10)


# Constants
EXCLUDE: set[str] = set(("survivability", "f_count", "fitness"))


# Simple instanciation.
# Define genomic library configuration & instanciate
gl_config: TableConfigNorm = gl_default_config()
gl_config["delete_db"] = True
glib: genomic_library = genomic_library(gl_config)
_logger.info("Genomic Library configured")

# Establish the Gene Pool
gp_config: GenePoolConfigNorm = gp_default_config()
gp: gene_pool = gene_pool({}, glib, gp_config)
_logger.info("Gene Pool configured")


def test_default_consistency() -> None:
    """Everything in the GPC should match the GL and GP"""
    for gpc_gc in gp.pool:
        _logger.debug(f"Checking GPC GC signature: {gpc_gc['signature'].data.hex()}")
        gl_gc: dict[str, Any] = glib[gpc_gc["signature"].data]
        gp_gc: dict[str, Any] = gp[gpc_gc["signature"].data]
        for field in filter(lambda x: not (x.endswith("_signature") or x in EXCLUDE), STORE_ALL_MEMBERS):
            if isinstance(gpc_gc[field], ndarray):
                if isinstance(gl_gc[field], memoryview):  # signatures
                    assert gpc_gc[field].tobytes() == gl_gc[field].tobytes()
                    assert gpc_gc[field].tobytes() == gp_gc[field].tobytes()
                else:
                    assert (gpc_gc[field] == gl_gc[field]).all()
                    assert (gpc_gc[field] == gp_gc[field]).all()
            elif isinstance(gpc_gc[field], _genetic_code) and gl_gc[field] is not None:
                assert gpc_gc[field]["signature"].tobytes() == gl_gc[field].tobytes()
                assert gpc_gc[field]["signature"].tobytes() == gp_gc[field].tobytes()
            elif isinstance(gpc_gc[field], _genetic_code) and gl_gc[field] is None:
                assert gpc_gc[field] is EMPTY_GENETIC_CODE
            elif field != "graph":
                assert gpc_gc[field] == gl_gc[field]
                assert gpc_gc[field] == gp_gc[field]


def test_dicts() -> None:
    """Test the dictionary methods"""
    for gpc_gc in gp.pool.dicts():
        gl_gc: dict[str, Any] = glib[gpc_gc["signature"]]
        gp_gc: dict[str, Any] = gp[gpc_gc["signature"]]
        for key, value in gpc_gc.items():
            if key in gl_gc:
                assert value == gl_gc[key], f"{key}: {value} != {gl_gc[key]}"
                assert isinstance(value, type(gl_gc[key])), f"{key}: {value} != {gl_gc[key]}"
            if key in gp_gc:
                assert value == gp_gc[key], f"{key}: {value} != {gp_gc[key]}"
                assert isinstance(value, type(gp_gc[key])), f"{key}: {value} != {gp_gc[key]}"


def test_dirty_update() -> None:
    """Test that an update to a member that will mark the GC as dirty gets the GC
    pushed back to the GP correctly."""
    for gpc_gc in gp.pool:
        gpc_gc["reference_count"] = 1975
        gpc_gc["evolvability"] = 0.78901234
        gpc_gc["e_count"] = 14
    gp.pool.purge()
    gp.pool.reset()
    gp.populate_local_cache()
    for gpc_gc in gp.pool:
        assert gpc_gc["reference_count"] == 1975
        assert gpc_gc["evolvability"] == 0.78901234
        assert gpc_gc["e_count"] == 14
        assert gpc_gc["reference_count"] == gp[gpc_gc["signature"].data]["reference_count"]
        assert gpc_gc["evolvability"] == gp[gpc_gc["signature"].data]["evolvability"]
        assert gpc_gc["e_count"] == gp[gpc_gc["signature"].data]["e_count"]


def test_two_gene_pools() -> None:
    """Test that two gene pools can be created and that they
    only interfere with each other in the right ways."""
    # Recreate GP (uses same config)
    gp_config1: GenePoolConfigNorm = gp_default_config()
    for v in gp_config1.values():
        v["delete_table"] = True
    gp1: gene_pool = gene_pool({}, glib, gp_config1)
    _logger.info("Gene Pool configured")
    for gpc_gc in gp1.pool:
        gpc_gc["reference_count"] = 1975
        gpc_gc["evolvability"] = 0.78901234
        gpc_gc["e_count"] = 14
    gp1.pool.purge()
    gp1.pool.reset()
    gp1.populate_local_cache()
    for gpc_gc in gp1.pool:
        assert gpc_gc["reference_count"] == 1975
        assert gpc_gc["evolvability"] == 0.78901234
        assert gpc_gc["e_count"] == 14
        assert gpc_gc["reference_count"] == gp1[gpc_gc["signature"].data]["reference_count"]
        assert gpc_gc["evolvability"] == gp1[gpc_gc["signature"].data]["evolvability"]
        assert gpc_gc["e_count"] == gp1[gpc_gc["signature"].data]["e_count"]

    # Create a second GP
    gp_config2: GenePoolConfigNorm = gp_default_config("2")
    gp2: gene_pool = gene_pool({}, glib, gp_config2)
    _logger.info("Gene Pool 2 configured")
    for gpc_gc in gp2.pool:
        assert gpc_gc["reference_count"] == 0
        assert gpc_gc["evolvability"] == 1.0
        assert gpc_gc["e_count"] == 1
        assert gpc_gc["reference_count"] == gp1[gpc_gc["signature"].data]["reference_count"]
        assert gpc_gc["evolvability"] == gp1[gpc_gc["signature"].data]["evolvability"]
        assert gpc_gc["e_count"] == gp1[gpc_gc["signature"].data]["e_count"]

    # Make sure the original GP is still correct
    for gpc_gc in gp1.pool:
        assert gpc_gc["reference_count"] == 1975
        assert gpc_gc["evolvability"] == 0.78901234
        assert gpc_gc["e_count"] == 14
        assert gpc_gc["reference_count"] == gp1[gpc_gc["signature"].data]["reference_count"]
        assert gpc_gc["evolvability"] == gp1[gpc_gc["signature"].data]["evolvability"]
        assert gpc_gc["e_count"] == gp1[gpc_gc["signature"].data]["e_count"]


if __name__ == "__main__":
    test_two_gene_pools()
    _logger.info("All tests passed")
