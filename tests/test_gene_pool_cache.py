import pytest
from egp_stores.gene_pool_cache import gene_pool_cache
from egp_types.xgc_validator import GGC_entry_validator
from surebrec.surebrec import generate, _logger as _surebrec_logger
from egp_types.gc_type_tools import is_pgc, ref_str
from logging import NullHandler, getLogger, INFO
from time import time
from pprint import pformat
from copy import deepcopy
from math import isclose
from numpy import ndarray
from typing import Any


_surebrec_logger.setLevel(INFO)
_logger = getLogger(__name__)
_logger.addHandler(NullHandler())


# Test data conformant with GGC schema
GGC_entry_validator.schema['inputs']['required'] = True
GGC_entry_validator.schema['outputs']['required'] = True
_TEST_DATA = {ggc['ref']: ggc for ggc in generate(GGC_entry_validator, 100, -1022196250, True)}


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

def dict_is_match(dct, gpc, removed=[]):
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
        gpc_ggc = gpc[dct_ggc['ref']]
        _logger.debug(f"Checking GGC ref {ref_str(gpc_ggc['ref'])} dct_ggc ref {ref_str(dct_ggc['ref'])}")
        assert dct_ggc['ref'] == gpc_ggc['ref'], f"Dictionary {ref_str(dct_ggc['ref'])} not in order in GPC!"
        for k, v in dct_ggc.items():
            if (k in gpc._gGC_cache.fields.keys() and not is_pgc(dct_ggc)) or (k in gpc._pGC_cache.fields.keys() and is_pgc(dct_ggc)):
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


def test_fill_element():
    gpc = gene_pool_cache(delta_size=4)
    for ggc in _TEST_DATA.values():
        _logger.debug(f"Adding entry {ref_str(ggc['ref'])}:\n{pformat(ggc, sort_dicts=True, indent=4)}")
        gpc[ggc['ref']] = ggc
    dict_is_match(_TEST_DATA, gpc)


def test_fill_update():
    gpc = gene_pool_cache(delta_size=4)
    gpc.update(_TEST_DATA)
    dict_is_match(_TEST_DATA, gpc)


def test_update():
    gpc = gene_pool_cache(delta_size=4)
    gpc.update({k:v for k, v in list(_TEST_DATA.items())[:int(len(_TEST_DATA) / 2)]})
    gpc.update({k:v for k, v in list(_TEST_DATA.items())[int(len(_TEST_DATA) / 2):]})
    dict_is_match(_TEST_DATA, gpc)

   
def test_delete():
    gpc = gene_pool_cache(delta_size=4)
    dct = deepcopy(_TEST_DATA)
    gpc.update(dct)
    removed = list(dct.keys())[::2]
    for ref in removed:
        del dct[ref]
        del gpc[ref]
    dict_is_match(dct, gpc, removed=removed)


def test_delete_assign():
    gpc = gene_pool_cache(delta_size=4)
    dct = deepcopy(_TEST_DATA)
    gpc.update(dct)
    removed = list(dct.keys())[::3]
    removed.reverse()
    for ref in removed:
        del dct[ref]
        del gpc[ref]
    for ref in removed[::2]:
        dct[ref] = _TEST_DATA[ref]
        gpc[ref] = _TEST_DATA[ref]
    dict_is_match(dct, gpc, removed=removed[1::2])


def test_delete_update():
    gpc = gene_pool_cache(delta_size=4)
    dct = deepcopy(_TEST_DATA)
    gpc.update(dct)
    removed = list(dct.keys())[::3]
    removed.reverse()
    for ref in removed:
        del dct[ref]
        del gpc[ref]
    update = {k:v for k, v in _TEST_DATA.items() if k in removed[::2]}
    dct.update(update)
    gpc.update(update)
    dict_is_match(dct, gpc, removed=removed[1::2])


@pytest.mark.skip(reason="Benchmerking value only.")
def test_write_performance():
    dct = {}
    gpc = gene_pool_cache()
    start = time()
    for _ in range(100):
        for k, v in _TEST_DATA.items():
            dct[k] = {_k: _v for _k, _v in v.items()}
    midway = time()
    for _ in range(100):
        for k, v in _TEST_DATA.items():
            gpc[k] = v
    stop = time()
    _logger.fatal(f"Creation times: dict: {midway - start:0.4f}s, GPC: {stop - midway:0.4f}s")


