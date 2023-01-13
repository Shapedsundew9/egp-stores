"""Genetic Material Store.

The GMS is a abstract base class for retrieving genetic codes.
"""

from random import randint
from logging import DEBUG, NullHandler, getLogger
from numpy.random import choice as weighted_choice
from random import choice
from json import load
from os.path import dirname, join


# Logging
_logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG = _logger.isEnabledFor(DEBUG)


# Constants
_GC_DEPTH = 'code_depth'
_CODON_DEPTH = 'codon_depth'
_GC_COUNT = 'num_codes'
_CODON_COUNT = 'num_codons'
_UNIQUE_GC_COUNT = 'num_unique_codes'
_UNIQUE_CODON_COUNT = 'num_unique_codons'
_OBJECT = 'object'
_ZERO_GC_COUNT = {_GC_COUNT: 0, _CODON_COUNT: 0, _GC_DEPTH: 0, _CODON_DEPTH: 0, _UNIQUE_GC_COUNT: 0, _UNIQUE_CODON_COUNT: 0}

# The update string
# No longer used but still here just in case of future need.
_WEIGHTED_VARIABLE_UPDATE_RAW = ('vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                   ', 0.0::REAL, 0::INTEGER)')
_WEIGHTED_FIXED_UPDATE_RAW = ('weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC})')
_FIXED_UPDATE_RAW = 'fixed_array_update({CSCC}, {PSCC}, {CSPC})'
_SCALAR_COUNT_UPDATE = '{CSCC} + {PSCC} - {CSPC}'
_WEIGHTED_SCALAR_UPDATE = '({CSCV} * {CSCC} + {PSCV} * {PSCC} - {CSPV} * {CSPC}) / ' + _SCALAR_COUNT_UPDATE
_PGC_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{pgc_evolvability}',
    'CSCC': 'EXCLUDED.{pgc_e_count}',
    'PSCV': '"__table__".{pgc_evolvability}',
    'PSCC': '"__table__".{pgc_e_count}',
    'CSPV': 'EXCLUDED.{_pgc_evolvability}',
    'CSPC': 'EXCLUDED.{_pgc_e_count}'
}
_PGC_EVO_UPDATE_STR = '{pgc_evolvability} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_E_COUNT_UPDATE_STR = '{pgc_e_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_FIT_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{pgc_fitness}',
    'CSCC': 'EXCLUDED.{pgc_f_count}',
    'PSCV': '"__table__".{pgc_fitness}',
    'PSCC': '"__table__".{pgc_f_count}',
    'CSPV': 'EXCLUDED.{_pgc_fitness}',
    'CSPC': 'EXCLUDED.{_pgc_f_count}'
}
_PGC_FIT_UPDATE_STR = '{pgc_fitness} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_PGC_F_COUNT_UPDATE_STR = '{pgc_f_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_EVO_UPDATE_MAP = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{e_count}',
    'PSCV': '"__table__".{evolvability}',
    'PSCC': '"__table__".{e_count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_e_count}'
}
_EVO_UPDATE_STR = '{evolvability} = ' + _WEIGHTED_SCALAR_UPDATE.format_map(_EVO_UPDATE_MAP)
_EVO_COUNT_UPDATE_STR = '{e_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_EVO_UPDATE_MAP)
_REF_UPDATE_MAP = {
    'CSCC': 'EXCLUDED.{reference_count}',
    'PSCC': '"__table__".{reference_count}',
    'CSPC': 'EXCLUDED.{_reference_count}'
}
_REF_UPDATE_STR = '{reference_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_REF_UPDATE_MAP)
UPDATE_STR = ','.join((
    '{updated} = NOW()',
    _PGC_EVO_UPDATE_STR,
    _PGC_E_COUNT_UPDATE_STR,
    _PGC_FIT_UPDATE_STR,
    _PGC_F_COUNT_UPDATE_STR,
    _EVO_UPDATE_STR,
    _EVO_COUNT_UPDATE_STR,
    _REF_UPDATE_STR))


# Data schema
with open(join(dirname(__file__), "formats/gms_table_format.json"), "r") as file_ptr:
    GMS_RAW_TABLE_SCHEMA = load(file_ptr)


class genetic_material_store():
    """Base class for all genetic material stores."""
    pass

"""
Some benchmarking on SHA256 generation
======================================
Python 3.8.5

>>> def a():
...     start = time()
...     for _ in range(10000000): int(sha256("".join(string.split()).encode()).hexdigest(), 16)
...     print(time() - start)
...
>>> a()
8.618626356124878
>>> def b():
...     start = time()
...     for _ in range(10000000): int.from_bytes(sha256("".join(string.split()).encode()).digest(), 'big')
...     print(time() - start)
...
>>> b()
7.211490631103516
>>> def c():
...     start = time()
...     for _ in range(10000000): sha256("".join(string.split()).encode()).hexdigest()
...     print(time() - start)
...
>>> c()
6.463267803192139
>>> def d():
...     start = time()
...     for _ in range(10000000): sha256("".join(string.split()).encode()).digest()
...     print(time() - start)
...
>>> d()
6.043259143829346
>>> def e():
...     start = time()
...     for _ in range(10000000): {sha256("".join(string.split()).encode()).digest(): "Test"}
...     print(time() - start)
...
>>> e()
6.640311002731323
>>> def f():
...     start = time()
...     for _ in range(10000000): {int.from_bytes(sha256("".join(string.split()).encode()).digest(), 'big'): "Test"}
...     print(time() - start)
...
>>> f()
7.6320412158966064
>>> def g():
...     start = time()
...     for _ in range(10000000): {sha256("".join(string.split()).encode()).hexdigest(): "Test"}
...     print(time() - start)
...
>>> g()
7.144319295883179
>>> def h1():
...     start = time()
...     for _ in range(10000000): getrandbits(256)
...     print(time() - start)
...
>>> h1()
1.0232288837432861
>>> def h2():
...     start = time()
...     for _ in range(10000000): getrandbits(128)
...     print(time() - start)
...
>>> h2()
0.8551476001739502
>>> def h3():
...     start = time()
...     for _ in range(10000000): getrandbits(64)
...     print(time() - start)
...
>>> h3()
0.764052152633667
>>> def i():
...     start = time()
...     for _ in range(10000000): getrandbits(256).to_bytes(32, 'big')
...     print(time() - start)
...
>>> i()
2.038336753845215
"""


"""
Some Benchmarking on hashing SHA256
===================================
Python 3.8.5

>>> a =tuple( (getrandbits(256).to_bytes(32, 'big') for _ in range(10000000)))
>>> b =tuple( (int(getrandbits(63)) for _ in range(10000000)))
>>> start = time(); c=set(a); print(time() - start)
1.8097834587097168
>>> start = time(); d=set(b); print(time() - start)
1.0908379554748535
"""
