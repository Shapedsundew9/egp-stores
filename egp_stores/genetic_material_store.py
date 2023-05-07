"""Genetic Material Store.

The GMS is a abstract base class for retrieving genetic codes.
"""

from json import load
from logging import DEBUG, NullHandler, getLogger, Logger
from os.path import dirname, join
from typing import Any

# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# Constants
_GC_DEPTH = 'code_depth'
_CODON_DEPTH = 'codon_depth'
_GC_COUNT = 'num_codes'
_CODON_COUNT = 'num_codons'
_UNIQUE_GC_COUNT = 'num_unique_codes'
_UNIQUE_CODON_COUNT = 'num_unique_codons'
_OBJECT = 'object'
_ZERO_GC_COUNT: dict[str, int] = {_GC_COUNT: 0, _CODON_COUNT: 0, _GC_DEPTH: 0, _CODON_DEPTH: 0, _UNIQUE_GC_COUNT: 0, _UNIQUE_CODON_COUNT: 0}

# The update string
# No longer used but still here just in case of future need.
_WEIGHTED_VARIABLE_UPDATE_RAW = ('vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}'
                                 ', 0.0::REAL, 0::INTEGER)')
_WEIGHTED_FIXED_UPDATE_RAW = 'weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC})'
_FIXED_UPDATE_RAW = 'fixed_array_update({CSCC}, {PSCC}, {CSPC})'
_SCALAR_COUNT_UPDATE = '{CSCC} + {PSCC} - {CSPC}'
_WEIGHTED_SCALAR_UPDATE = '({CSCV} * {CSCC} + {PSCV} * {PSCC} - {CSPV} * {CSPC}) / ' + _SCALAR_COUNT_UPDATE
_PGC_EVO_UPDATE_MAP: dict[str, str] = {
    'CSCV': 'EXCLUDED.{pgc_evolvability}',
    'CSCC': 'EXCLUDED.{pgc_e_count}',
    'PSCV': '"__table__".{pgc_evolvability}',
    'PSCC': '"__table__".{pgc_e_count}',
    'CSPV': 'EXCLUDED.{_pgc_evolvability}',
    'CSPC': 'EXCLUDED.{_pgc_e_count}'
}
_PGC_EVO_UPDATE_STR: str = '{pgc_evolvability} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_E_COUNT_UPDATE_STR: str = '{pgc_e_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_FIT_UPDATE_MAP: dict[str, str] = {
    'CSCV': 'EXCLUDED.{pgc_fitness}',
    'CSCC': 'EXCLUDED.{pgc_f_count}',
    'PSCV': '"__table__".{pgc_fitness}',
    'PSCC': '"__table__".{pgc_f_count}',
    'CSPV': 'EXCLUDED.{_pgc_fitness}',
    'CSPC': 'EXCLUDED.{_pgc_f_count}'
}
_PGC_FIT_UPDATE_STR: str = '{pgc_fitness} = ' + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_PGC_F_COUNT_UPDATE_STR: str = '{pgc_f_count} = ' + _FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_EVO_UPDATE_MAP: dict[str, str] = {
    'CSCV': 'EXCLUDED.{evolvability}',
    'CSCC': 'EXCLUDED.{e_count}',
    'PSCV': '"__table__".{evolvability}',
    'PSCC': '"__table__".{e_count}',
    'CSPV': 'EXCLUDED.{_evolvability}',
    'CSPC': 'EXCLUDED.{_e_count}'
}
_EVO_UPDATE_STR: str = '{evolvability} = ' + _WEIGHTED_SCALAR_UPDATE.format_map(_EVO_UPDATE_MAP)
_EVO_COUNT_UPDATE_STR: str = '{e_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_EVO_UPDATE_MAP)
_REF_UPDATE_MAP: dict[str, str] = {
    'CSCC': 'EXCLUDED.{reference_count}',
    'PSCC': '"__table__".{reference_count}',
    'CSPC': 'EXCLUDED.{_reference_count}'
}
_REF_UPDATE_STR: str = '{reference_count} = ' + _SCALAR_COUNT_UPDATE.format_map(_REF_UPDATE_MAP)
UPDATE_STR: str = ','.join((
    '{updated} = NOW()',
    _PGC_EVO_UPDATE_STR,
    _PGC_E_COUNT_UPDATE_STR,
    _PGC_FIT_UPDATE_STR,
    _PGC_F_COUNT_UPDATE_STR,
    _EVO_UPDATE_STR,
    _EVO_COUNT_UPDATE_STR,
    _REF_UPDATE_STR))


# Data schema
with open(join(dirname(__file__), "formats/gms_table_format.json"), "r", encoding="utf-8") as file_ptr:
    GMS_RAW_TABLE_SCHEMA: dict[str, Any] = load(file_ptr)


class genetic_material_store():
    """Base class for all genetic material stores."""
