"""Test the Gene Pool."""
from logging import INFO, Logger, NullHandler, getLogger
from typing import Any
from numpy import ndarray

from pypgtable.pypgtable_typing import TableConfigNorm

from egp_stores.egp_typing import GenePoolConfigNorm
from egp_stores.gene_pool import default_config as gp_default_config
from egp_stores.gene_pool import gene_pool
from egp_stores.genomic_library import default_config as gl_default_config
from egp_stores.genomic_library import genomic_library
from egp_stores.gene_pool_cache import STORE_ALL_MEMBERS, _genetic_code


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
getLogger("pypgtable").setLevel(INFO)
getLogger("obscure_password").setLevel(INFO)


EXCLUDE: set[str] = set(("survivability", "f_count", "fitness"))


def test_default_consistency() -> None:
    """Simple instanciation."""
    # Define genomic library configuration & instanciate
    gl_config: TableConfigNorm = gl_default_config()
    gl_config["delete_db"] = True
    glib: genomic_library = genomic_library(gl_config)
    _logger.info("Genomic Library configured")

    # Establish the Gene Pool
    gp_config: GenePoolConfigNorm = gp_default_config()
    gp: gene_pool = gene_pool({}, glib, gp_config)
    _logger.info("Gene Pool configured")

    # Everything in the GPC should match the GL and GP
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
            elif isinstance(gpc_gc[field], _genetic_code):
                assert gpc_gc[field]["signature"].tobytes() == gl_gc[field].tobytes()
                assert gpc_gc[field]["signature"].tobytes() == gp_gc[field].tobytes()
            else:
                assert gpc_gc[field] == gl_gc[field]
                assert gpc_gc[field] == gp_gc[field]
