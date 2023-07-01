import pytest

from copy import deepcopy
from logging import INFO, Logger, NullHandler, getLogger
from math import isclose
from pprint import pformat
from sys import getsizeof
from time import time
from typing import Any
from random import random
from os.path import dirname, join
from json import load
from numpy import ndarray

from egp_types.gc_type_tools import is_pgc, PHYSICAL_PROPERTY
from egp_types.reference import ref_str
from egp_types.xgc_validator import gGC_entry_validator
from egp_types.xGC import xGC
from pympler.asizeof import asizeof
from surebrec.surebrec import _logger as _surebrec_logger
from surebrec.surebrec import generate

from egp_stores.gene_pool_cache import gene_pool_cache

_surebrec_logger.setLevel(INFO)
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


# Use surebrec random data generation. This is a lot slower than using the predefined test data from simple_gl.json
# It is also less robust due to the blending of pGC & gGC schemas into one.
USE_SUREBREC = False


def choose_p(ggc: dict[str, Any]) -> dict[str, Any]:
    """Corrects a randomly generated GGC entry to be conformant with the pGC schema 50% of the time"""
    if random() < 0.5:
        for key in filter(lambda x: 'pgc_' in x, ggc.keys()):
            ggc[key] = None
    elif ggc.get('input_types', []) and ggc.get('output_types', []):
        if -3 not in ggc['input_types']:
            ggc['input_types'][0] = -3
        if -3 not in ggc['output_types']:
            ggc['output_types'][0] = -3
        ggc['properties'] = ggc['properties'] | PHYSICAL_PROPERTY
    return ggc

# Test data conformant with GGC schema
gGC_entry_validator.schema['inputs']['required'] = True
gGC_entry_validator.schema['outputs']['required'] = True

_TEST_DATA: dict[int, dict[str, Any]]
if USE_SUREBREC:
    _TEST_DATA = {ggc['ref']: choose_p(ggc) for ggc in generate(gGC_entry_validator, 100, -1022196250, True)}
else:
    with open(join(dirname(__file__), 'data/simple_gl.json'), 'r', encoding='utf-8') as fptr:
        # FIXME: simple_gl is in GL entry format & we need to convert to GP entry format
        _TEST_DATA = {ggc['ref']: ggc for ggc in load(fptr)}


def element_is_match(a: Any, b: Any) -> bool:
    """Custom matching function.

    None cannot be represented in an ndarray. In this case the
    ndarray will return the default value (0).
    Also does float 'isclose' equality.

    Args
    ----
    a: Test data value
    b: GPC value

    Returns
    -------
    True if a and b match.
    """
    if isinstance(a, float) or isinstance(b, float):
        return isclose(a, b, rel_tol=1e-06)
    if a is None and b is None:
        return True
    if a is None and isinstance(b, ndarray):
        return not b.sum()
    if a is None and b == 0:
        return True
    return a == b


def dict_is_match(dct, gpc: gene_pool_cache, removed: list[int] = []) -> None:
    assert len(dct) == len(gpc), f"Length of dictionary ({len(dct)}) != length of GPC ({len(gpc)})!"
    for dct_ref in dct.keys():
        assert dct_ref in gpc, f"Dictionary {ref_str(dct_ref)} not in GPC!"
    assert set(dct.keys()) == set(gpc.keys()), "Dictionary & GPC sets of keys() do not match!"
    assert set((v['pgc_ref'] for v in dct.values())) == set((v['pgc_ref'] for v in gpc.values())), (
        "Dictionary & GPC sets of values() do not match!")
    assert set(dct.keys()) == set((k for k, _ in gpc.items())), "GPC items() keys do not match dictionary keys()."
    assert set((v['pgc_ref'] for v in dct.values())) == set((v['pgc_ref'] for _, v in gpc.items())), (
        "Dictionary & GPC sets of values from items do not match!")
    for dct_ggc in dct.values():
        gpc_ggc: xGC = gpc[dct_ggc['ref']]
        _logger.debug(f"Checking GGC ref {ref_str(gpc_ggc['ref'])} dct_ggc ref {ref_str(dct_ggc['ref'])}")
        assert dct_ggc['ref'] == gpc_ggc['ref'], f"Dictionary {ref_str(dct_ggc['ref'])} not in order in GPC!"
        for k, v in dct_ggc.items():
            if (k in gpc._ggc_cache.fields and not is_pgc(dct_ggc)) or (k in gpc._pgc_cache.fields and is_pgc(dct_ggc)):
                if isinstance(v, list) or isinstance(v, ndarray):
                    for i, j in zip(v, gpc[dct_ggc['ref']][k]):
                        if not element_is_match(i, j):
                            _logger.debug(f"'{k}' does not match: Test data element {i} != GPC element {j}.")
                            _logger.debug(f"'{k}' test data: {v}, GPC data: {gpc[dct_ggc['ref']][k]}.")
                            assert element_is_match(i, j)
                else:
                    if not element_is_match(v, gpc[dct_ggc['ref']][k]):
                        _logger.debug(f"'{k}' test data: {v}, GPC data: {gpc[dct_ggc['ref']][k]}.")
                        assert element_is_match(v, gpc[dct_ggc['ref']][k])
        for ref in removed:
            assert ref not in gpc, f"Dictionary ref {ref_str(ref)} should not be in GPC!"


def test_fill_element() -> None:
    gpc: gene_pool_cache = gene_pool_cache(delta_size=4)
    for ggc in _TEST_DATA.values():
        _logger.debug(f"Adding entry {ref_str(ggc['ref'])}:\n{pformat(ggc, sort_dicts=True, indent=4)}")
        gpc[ggc['ref']] = ggc
    _logger.info(f"Dict size: sys.getsizeof = {getsizeof(_TEST_DATA)} bytes, pympler.asizeof = {asizeof(_TEST_DATA)} bytes.")
    _logger.info(f"GPC size: sys.getsizeof = {getsizeof(gpc)} bytes, pympler.asizeof = {asizeof(gpc)} bytes.")
    # 14:01:30 INFO test_gene_pool_cache.py 93 Dict size: sys.getsizeof = 4688 bytes, pympler.asizeof = 5399488 bytes.
    # 14:01:30 INFO test_gene_pool_cache.py 94 GPC size: sys.getsizeof = 56 bytes, pympler.asizeof = 204576 bytes.
    dict_is_match(_TEST_DATA, gpc)


def test_fill_update() -> None:
    gpc: gene_pool_cache = gene_pool_cache(delta_size=4)
    gpc.update(_TEST_DATA)
    dict_is_match(_TEST_DATA, gpc)


def test_update() -> None:
    gpc: gene_pool_cache = gene_pool_cache(delta_size=4)
    gpc.update({k: v for k, v in list(_TEST_DATA.items())[:int(len(_TEST_DATA) / 2)]})
    gpc.update({k: v for k, v in list(_TEST_DATA.items())[int(len(_TEST_DATA) / 2):]})
    dict_is_match(_TEST_DATA, gpc)


def test_delete() -> None:
    gpc: gene_pool_cache = gene_pool_cache(delta_size=4)
    dct: dict[int, dict[str, Any]] = deepcopy(_TEST_DATA)
    gpc.update(dct)
    removed: list[int] = list(dct.keys())[::2]
    for ref in removed:
        del dct[ref]
        del gpc[ref]
    dict_is_match(dct, gpc, removed=removed)


def test_delete_assign() -> None:
    gpc: gene_pool_cache = gene_pool_cache(delta_size=4)
    dct: dict[int, dict[str, Any]] = deepcopy(_TEST_DATA)
    gpc.update(dct)
    removed: list[int] = list(dct.keys())[::3]
    removed.reverse()
    for ref in removed:
        del dct[ref]
        del gpc[ref]
    for ref in removed[::2]:
        dct[ref] = _TEST_DATA[ref]
        gpc[ref] = _TEST_DATA[ref]
    dict_is_match(dct, gpc, removed=removed[1::2])


def test_delete_update() -> None:
    gpc: gene_pool_cache = gene_pool_cache(delta_size=4)
    dct: dict[int, dict[str, Any]] = deepcopy(_TEST_DATA)
    gpc.update(dct)
    removed: list[int] = list(dct.keys())[::3]
    removed.reverse()
    for ref in removed:
        del dct[ref]
        del gpc[ref]
    update: dict[int, dict[str, Any]] = {k: v for k, v in _TEST_DATA.items() if k in removed[::2]}
    dct.update(update)
    gpc.update(update)
    dict_is_match(dct, gpc, removed=removed[1::2])


@pytest.mark.skip(reason="Benchmarking value only.")
def test_write_performance() -> None:
    dct: dict[int, dict[str, Any]] = {}
    gpc: gene_pool_cache = gene_pool_cache()
    start: float = time()
    for _ in range(100):
        for k, v in _TEST_DATA.items():
            dct[k] = {_k: _v for _k, _v in v.items()}
    midway: float = time()
    for _ in range(100):
        for k, v in _TEST_DATA.items():
            gpc[k] = v
    stop: float = time()
    _logger.fatal(f"Creation times: dict: {midway - start:0.4f}s, GPC: {stop - midway:0.4f}s")
