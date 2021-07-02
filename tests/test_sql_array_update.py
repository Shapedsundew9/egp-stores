"""Validate the SQL function array_update()."""

from random import random, randint
from numpy import array, float64, int64
from itertools import product
from pypgtable import table
from os.path import join, dirname
from pytest import approx


_SQL_STR = 'array_update({cscv}, {cscc}, {pscv}, {pscc}, {cspv}, {cspc}, {fp1}, {i1}), {tv}, {tc}'
_LITERALS = {'fp1': 1.0, 'i1': 1}
_CONFIG = {
    'database': {
        'dbname': 'test_db'
    },
    'table': 'test_table',
    'schema': {
        'cscv': {
            'type': 'DOUBLE PRECISION[]'
        },
        'cscc': {
            'type': 'BIGINT[]'
        },
        'pscv': {
            'type': 'DOUBLE PRECISION[]'
        },
        'pscc': {
            'type': 'BIGINT[]'
        },
        'cspv': {
            'type': 'DOUBLE PRECISION[]'
        },
        'cspc': {
            'type': 'BIGINT[]'
        },
        'tv': {
            'type': 'DOUBLE PRECISION[]'
        },
        'tc': {
            'type': 'BIGINT[]'
        }
    },
    'delete_db': False,
    'delete_table': True,
    'create_db': True,
    'create_table': True,
    'wait_for_db': False,
    'wait_for_table': False
}


def _random_pair(length):
    """Generate a value & count pair of arrays.

    Value array is populated with flaoting point values 0.0 <= x < 1.0
    Count array is populated 1 <= x <= 100000

    Args
    ----
    length (int): The length of the arrays to return.

    Returns
    -------
    ((list(float), list(int)))
    """
    return [random() for _ in range(length)], [randint(1, 100000) for _ in range(length)]


def _random_set(lengths):
    """Generate a full set of array parameters based on the lengths provided.

    The arrays are returned as ((cscv, cscc), (pscv, pscc), (cspv, cspc))
    Value arrays are populated with flaoting point values 0.0 <= x < 1.0
    Count arrays are populated 1 <= x <= 100000

    Args
    ----
    lengths ((int, int, int)): csc, psc & csp array pair lengths respectively.

    Returns
    -------
    ((list(float), list(int)), (list(float), list(int)), (list(float), list(int)))
    """
    return _random_pair(lengths[0]), _random_pair(lengths[1]), _random_pair(lengths[2])


def _random_length(criteria, csp_len):
    """Generate a length meeting the criteria.

    > return a length greater than csp_len (max csp_len * 2)
    < return a length less than csp_len
    = return a length == to csp_len

    In the case csp_len == 1 and criteria '<' a length of 1 will be returned.

    Args
    ----
    criteria (str): where str is one of '<', '>' or '='

    Returns
    -------
    (int): A length meeting the criteria.
    """
    if criteria == '<':
        x_len = csp_len - randint(1, csp_len)
        if not x_len: x_len = 1
    elif criteria == '>':
        x_len = csp_len + randint(1, csp_len)
    else:
        x_len = csp_len
    return x_len


def _random_lengths(criteria):
    """Generate a tuple of random lengths meeting the criteria.

    The maximum length of any array is 200.

    Args
    ----
    criteria ((str, str)): where str is one of '<', '>' or '='

    Returns
    -------
    ((int, int, int)): Lengths meeting the criteria
    """
    csp_len = randint(1, 100)
    return _random_length(criteria[0], csp_len), _random_length(criteria[1], csp_len), csp_len


def _combo_generator():
    """Generate all combinations of random parameter array lengths.

    The update_array() SQL function takes 3 pairs of variable length arrays:
        csc: cscv & cscc
        psc: pscv & pscc
        csp: cspv & cspc
    Where:
        '<' means shorter than csp.
        '>' means longer than csp.
        '=' means the same length as csp
    All the possible combinations of pair lengths are:
        (csc, psc)
        ----------
        ('<', '<')
        ('<', '>')
        ('<', '=')
        ('>', '<')
        ('>', '>')
        ('>', '=')
        ('=', '<')
        ('=', '>')
        ('=', '=')

    This generator returns a tuple (cscv, cscc, pscv, pscc, cspv, cspc) meeting the
    length constraints of each combination in turn, looping infinitely.
    The array values are random and valid.

    Returns
    -------
    ((list(float), list(int)), (list(float), list(int)), (list(float), list(int)))
    """
    while True:
        for combo in product('<>=', repeat=2):
            yield _random_set(_random_lengths(combo))


def _expected_result(array_set):
    cscv, cscc = array_set[0]
    pscv, pscc = array_set[1]
    cspv, cspc = array_set[2]

    max_len = max((len(cscv), len(cscc), len(pscv), len(pscc), len(cspv), len(cspc)))
    while len(cscv) < max_len: cscv.append(1.0)
    while len(cscc) < max_len: cscc.append(1)
    while len(pscv) < max_len: pscv.append(1.0)
    while len(pscc) < max_len: pscc.append(1)
    while len(cspv) < max_len: cspv.append(1.0)
    while len(cspc) < max_len: cspc.append(1)

    cscv = array(cscv, dtype=float64)
    cscc = array(cscc, dtype=int64)
    pscv = array(pscv, dtype=float64)
    pscc = array(pscc, dtype=int64)
    cspv = array(cspv, dtype=float64)
    cspc = array(cspc, dtype=int64)

    return list((cscv * cscc + pscv * pscc - cspv * cspc) / (cscc + pscc - cspc))


def _generate_test_case():
    input_generator =  _combo_generator()
    while True:
        inputs = next(input_generator)
        yield inputs, _expected_result(inputs)


def _create_testcases(n):
    testcase_generator = _generate_test_case()
    testcases = []
    for _ in range(10000):
        inputs, result = next(testcase_generator)
        testcases.append(
            {
                'cscv': inputs[0][0],
                'cscc': inputs[0][1],
                'pscv': inputs[1][0],
                'pscc': inputs[1][1],
                'cspv': inputs[2][0],
                'cspc': inputs[2][1],
                'tv': result[0],
                'tc': result[1]
            }
        )
    return testcases


def test_sql_array_update():
    t = table(_CONFIG)
    with open(join(dirname(__file__), '../data/array_functions.sql'), 'r') as fileptr:
        t.arbitrary_sql(fileptr.read())
    t.insert(_create_testcases(10000))
    for row in t.select(_SQL_STR, _LITERALS):
        print(row)
        assert(row[0] == approx(row[2]))
        assert(row[1] == row[3])


if __name__ == '__main__':
    testcase_generator = _generate_test_case()
    for _ in range(20):
        print("Test case: ", next(testcase_generator))