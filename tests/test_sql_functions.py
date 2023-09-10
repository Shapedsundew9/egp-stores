"""Validate the SQL functions used by the genomic library."""

# TODO: Add overflow & underflow test cases.

from copy import copy, deepcopy
from json import load
from os.path import dirname, join
from random import randint, random
from typing import Any, Literal, Generator

from numpy import all, array, float64, int64
from numpy.typing import NDArray
from pypgtable import table
from pytest import approx

VCpair = tuple[list[float], list[int]]
VCset = tuple[VCpair, VCpair, VCpair]

_DEFAULT_VALUE: float = 1.0
_DEFAULT_COUNT: Literal[1] = 1
_SQL_STR: str = (
    "SELECT weighted_fixed_array_update({cscv}::REAL[], {cscc}::INT[], {pscv}::REAL[], {pscc}::INT[], {cspv}::REAL[], {cspc}::INT[]),"
    " fixed_array_update({cscc}::INT[], {pscc}::INT[], {cspc}::INT[]),"
    " {tv}, {tc} FROM test_table"
)
_CONFIG: dict[str, Any] = {
    "database": {"dbname": "test_db"},
    "table": "test_table",
    "schema": {
        "idx": {"type": "INTEGER"},
        "cscv": {"type": "REAL[]"},
        "cscc": {"type": "INT[]"},
        "pscv": {"type": "REAL[]"},
        "pscc": {"type": "INT[]"},
        "cspv": {"type": "REAL[]"},
        "cspc": {"type": "INT[]"},
        "tv": {"type": "REAL[]"},
        "tc": {"type": "INT[]"},
    },
    "data_files": [],
    "delete_db": False,
    "delete_table": True,
    "create_db": True,
    "create_table": True,
    "wait_for_db": False,
    "wait_for_table": False,
}

with open(
    join(dirname(__file__), "../egp_stores/formats/meta_table_format.json"), "r"
) as file_ptr:
    _META_TABLE_SCHEMA = load(file_ptr)


def _random_pair(length, base: VCpair | None = None) -> VCpair:
    """Generate a value & count pair of arrays.

    Value array is populated with floating point values 0.0 <= x < 1.0
    If base is defined value is scaled so that the average increment
    to value per count <= 1.0.
    Count array is populated 1 <= x <= 50000 if base_count is None
    If base is defined each value is incremented by 1 <= x <= 50000

    Args
    ----
    length: The length of the arrays to return.
    base): Base value, count arrays

    Returns
    -------
    Values, counts
    """
    if base is None:
        values: list[float] = [random() for _ in range(length)]
        variable_vector_weights_update: list[int] = [
            randint(1, 50000) for _ in range(length)
        ]
    else:
        counts: list[int] = [randint(1, 50000) for _ in base[1]]
        variable_vector_weights_update = [b + c for c, b in zip(counts, base[1])]
        values = [
            v + (((1 - v) * random()) * c) / (b + c)
            for c, v, b in zip(counts, base[0], base[1])
        ]
        variable_vector_weights_update.extend(
            [_DEFAULT_COUNT] * (len(base[0]) - length)
        )
        values.extend([_DEFAULT_VALUE] * (len(base[0]) - length))
    return values, variable_vector_weights_update


def _random_set(lengths: tuple[int, int, int]) -> tuple[VCpair, VCpair, VCpair]:
    """Generate a full set of array parameters based on the lengths provided.

    The arrays are returned as ((cscv, cscc), (pscv, pscc), (cspv, cspc))
    Value arrays are populated with floating point values 0.0 <= x < 1.0
    Count arrays are populated 1 <= x <= 100000
    There are constraints on the lengths & counts: cspc <= pscc and cspc <= cscc

    Args
    ----
    lengths : csc, psc & csp array pair lengths respectively.

    Returns
    -------
    Array paramters
    """
    csp: VCpair = _random_pair(lengths[0])
    base_pscc: list[int] = copy(csp[1])
    base_cscc: list[int] = copy(csp[1])
    base_pscv: list[float] = copy(csp[0])
    base_cscv: list[float] = copy(csp[0])
    base_pscc.extend([_DEFAULT_COUNT] * (lengths[1] - lengths[0]))
    base_cscc.extend([_DEFAULT_COUNT] * (lengths[2] - lengths[0]))
    base_pscv.extend([_DEFAULT_VALUE] * (lengths[1] - lengths[0]))
    base_cscv.extend([_DEFAULT_VALUE] * (lengths[2] - lengths[0]))
    return (
        _random_pair(lengths[0], (base_cscv, base_cscc)),
        _random_pair(lengths[0], (base_pscv, base_pscc)),
        csp,
    )


def _random_length(
    criteria: Literal["<"] | Literal[">"] | Literal["="], csp_len: int
) -> int:
    """Generate a length meeting the criteria.

    > return a length greater than csp_len (max csp_len * 2)
    < return a length less than csp_len
    = return a length == to csp_len

    In the case csp_len == 1 and criteria '<' a length of 1 will be returned.

    Args
    ----
    criteria: One of '<', '>' or '='

    Returns
    -------
    A length meeting the criteria.
    """
    if criteria == "<":
        x_len: int = csp_len - randint(1, csp_len)
        if not x_len:
            x_len = 1
    elif criteria == ">":
        x_len = csp_len + randint(1, csp_len)
    else:
        x_len = csp_len
    return x_len


def _random_lengths(
    criteria: tuple[Literal["<", ">", "="], Literal["<", ">", "="]]
) -> tuple[int, int, int]:
    """Generate a tuple of random lengths meeting the criteria.

    The maximum length of any array is 200.

    Args
    ----
    criteria ((str, str)): where str is one of '<', '>' or '='

    Returns
    -------
    ((int, int, int)): Lengths meeting the criteria
    """
    csp_len: int = randint(1, 100)
    # Mixed length arrays no longer supported
    # return _random_length(criteria[0], csp_len), _random_length(criteria[1], csp_len), csp_len
    return csp_len, csp_len, csp_len


def _combo_generator() -> Generator[tuple[VCpair, VCpair, VCpair], None, None]:
    """Generate all combinations of random parameter array lengths.

    The update_array() SQL function takes 3 pairs of variable length arrays:
        csc: cscv & cscc
        psc: pscv & pscc
        csp: cspv & cspc
    Where:
        '>' means longer than csp.
        '=' means the same length as csp
    NB: psc and csc cannot be shorter than csp.
    All the possible combinations of pair lengths are:
        (csc, psc)
        ----------
        ('<', '<')
        ('<', '=')
        ('=', '<')
        ('=', '=')
    This generator returns a tuple (cscv, cscc, pscv, pscc, cspv, cspc) meeting the
    length constraints of each combination in turn, looping infinitely.
    The array values are random and valid.

    Returns
    -------
    ((list(float), list(int)), (list(float), list(int)), (list(float), list(int)))
    """
    while True:
        for combo in (("<", "<"), ("<", "="), ("=", "<"), ("=", "=")):
            yield _random_set(_random_lengths(combo))


def _expected_result(array_set: tuple[VCpair, VCpair, VCpair]) -> VCpair:
    """Calculate the result we should expect.

    Input data is sanity checked.

    Args
    ----
    array_set (see below): Set of arrays.
            cscv, cscc = array_set[0]
            pscv, pscc = array_set[1]
            cspv, cspc = array_set[2]

    Returns
    -------
    (list(float), list(int)): tv, tc
    """
    cscv: list[float] = array_set[0][0]
    cscc: list[int] = array_set[0][1]
    pscv: list[float] = array_set[1][0]
    pscc: list[int] = array_set[1][1]
    cspv: list[float] = array_set[2][0]
    cspc: list[int] = array_set[2][1]

    max_len: int = max(
        (len(cscv), len(cscc), len(pscv), len(pscc), len(cspv), len(cspc))
    )
    while len(cscv) < max_len:
        cscv.append(1.0)
    while len(cscc) < max_len:
        cscc.append(1)
    while len(pscv) < max_len:
        pscv.append(1.0)
    while len(pscc) < max_len:
        pscc.append(1)
    while len(cspv) < max_len:
        cspv.append(1.0)
    while len(cspc) < max_len:
        cspc.append(1)

    _cscv: NDArray[float64] = array(cscv, dtype=float64)
    _cscc: NDArray[int64] = array(cscc, dtype=int64)
    _pscv: NDArray[float64] = array(pscv, dtype=float64)
    _pscc: NDArray[int64] = array(pscc, dtype=int64)
    _cspv: NDArray[float64] = array(cspv, dtype=float64)
    _cspc: NDArray[int64] = array(cspc, dtype=int64)

    tw: NDArray[float64] = _cscv * _cscc + _pscv * _pscc - _cspv * _cspc
    tc: NDArray[int64] = _cscc + _pscc - _cspc
    tv: NDArray[float64] = tw / tc

    # Sanity
    if not (all(tc > 0) and all(tw >= 0.0) and all(tv >= 0.0) and all(tv <= 1.0)):
        print("cscv: ", _cscv)
        print("cscc: ", _cscc)
        print("pscv: ", _pscv)
        print("pscc: ", _pscc)
        print("cspv: ", _cspv)
        print("cspc: ", _cspc)
        print("tw: ", tw)
        print("tv: ", tv)
        print("tc: ", tc)
        assert all(tc > 0)
        assert all(tw >= 0.0)
        assert all(tv >= 0.0)
        assert all(tv <= 1.0)

    return [float(v) for v in tv], [int(c) for c in tc]


def _generate_test_case() -> Generator[tuple[VCset, VCpair], None, None]:
    """Test data generator.

    Return a single test case as a tuple of inputs and results.

    Returns
    -------
    inputs (): (cscv, cscc, pscv, pscc, cspv, cspc)
    results (): (tv, tc)
    """
    input_generator: Generator[VCset, None, None] = _combo_generator()
    while True:
        inputs: VCset = next(input_generator)
        yield inputs, _expected_result(deepcopy(inputs))


def _create_testcases(n):
    """Genetrate a list of test cases.

    Args
    ----
    n (int): Number of test cases to generate.

    Returns
    -------
    list(dict): Output of _generate_test_case()
         in a dictionary format.
    """
    testcase_generator: Generator[
        tuple[VCset, VCpair], None, None
    ] = _generate_test_case()
    testcases: list[dict[str, Any]] = []
    for idx in range(n):
        testcase: tuple[VCset, VCpair] = next(testcase_generator)
        inputs: VCset = testcase[0]
        result: VCpair = testcase[1]
        testcases.append(
            {
                "idx": idx,
                "cscv": inputs[0][0],
                "cscc": inputs[0][1],
                "pscv": inputs[1][0],
                "pscc": inputs[1][1],
                "cspv": inputs[2][0],
                "cspc": inputs[2][1],
                "tv": result[0],
                "tc": result[1],
            }
        )
    return testcases


def meta_table_config(config: dict[str, Any], create: bool = False) -> dict[str, Any]:
    """Create the meta table config from the genomic library config.

    Args
    ----
    config (dict): pypgtable config of the genomic library.
    create (bool): If true (re)create the meta data table.

    Returns
    -------
    (dict): pypgtable config of the genomic library meta table.
    """
    _config: dict[str, Any] = {
        "table": config["table"] + "_meta",
        "schema": _META_TABLE_SCHEMA,
        "database": deepcopy(config["database"]),
    }
    if create:
        _config["delete_table"] = True
        _config["create_table"] = True
    return _config


def test_sql_array_update() -> None:
    """Validate the SQL functions match the model."""

    meta: table = table(meta_table_config(_CONFIG, create=True))
    t: table = table(_CONFIG)
    if t.raw.creator:
        meta = table(meta_table_config(_CONFIG, create=True))
        with open(
            join(dirname(__file__), "../egp_stores/data/gl_functions.sql"), "r"
        ) as fileptr:
            sql_text: str = fileptr.read()
            sql_text = sql_text.replace("__meta_table_name__", meta.raw.config["table"])
            sql_text = sql_text.replace("__gl_table_name__", t.raw.config["table"])
            t.raw.arbitrary_sql(sql_text, read=False)
    t.insert(_create_testcases(300))
    for row in t.raw.arbitrary_sql(_SQL_STR):
        assert row[0] == approx(row[2])
        assert row[1] == row[3]
