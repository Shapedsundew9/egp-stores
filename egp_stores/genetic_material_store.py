"""Genetic Material Store.

The GMS is a abstract base class for retrieving genetic codes.
It defines common constants and methods for all GMSs.

The GC ancestory graph: A graph of the GCs ancestory. The GC ancestory graph is a directed graph with the GCs as nodes.
The GC structure graph: A graph of the GCs structure. The GC structure graph is a directed graph with the GCs as nodes.

Each graph is a view of the same base graph with edges labelled as ancestors or GC's.
"""

from json import load
from logging import DEBUG, NullHandler, getLogger, Logger
from os.path import dirname, join
from typing import Any, Callable, Literal
from functools import lru_cache

from pypgtable.table import table
from pypgtable.pypgtable_typing import RowIter, TableConfig, TableConfigNorm, PtrMap


# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# The update string
# No longer used but still here just in case of future need.
# _WEIGHTED_VARIABLE_UPDATE_RAW = (
#     "vector_weighted_variable_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC}" ", 0.0::REAL, 0::INTEGER)"
# )
_WEIGHTED_FIXED_UPDATE_RAW = "weighted_fixed_array_update({CSCV}, {CSCC}, {PSCV}, {PSCC}, {CSPV}, {CSPC})"
_FIXED_UPDATE_RAW = "fixed_array_update({CSCC}, {PSCC}, {CSPC})"
_SCALAR_COUNT_UPDATE = "({CSCC} + {PSCC} - {CSPC})"
_WEIGHTED_SCALAR_UPDATE = "({CSCV} * {CSCC} + {PSCV} * {PSCC} - {CSPV} * {CSPC}) / " + _SCALAR_COUNT_UPDATE
_PGC_EVO_UPDATE_MAP: dict[str, str] = {
    "CSCV": "EXCLUDED.{pgc_evolvability}",
    "CSCC": "EXCLUDED.{pgc_e_count}",
    "PSCV": '"__table__".{pgc_evolvability}',
    "PSCC": '"__table__".{pgc_e_count}',
    "CSPV": "EXCLUDED.{_pgc_evolvability}",
    "CSPC": "EXCLUDED.{_pgc_e_count}",
}
_PGC_EVO_UPDATE_STR: str = "{pgc_evolvability} = " + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_E_COUNT_UPDATE_STR: str = "{pgc_e_count} = " + _FIXED_UPDATE_RAW.format_map(_PGC_EVO_UPDATE_MAP)
_PGC_FIT_UPDATE_MAP: dict[str, str] = {
    "CSCV": "EXCLUDED.{pgc_fitness}",
    "CSCC": "EXCLUDED.{pgc_f_count}",
    "PSCV": '"__table__".{pgc_fitness}',
    "PSCC": '"__table__".{pgc_f_count}',
    "CSPV": "EXCLUDED.{_pgc_fitness}",
    "CSPC": "EXCLUDED.{_pgc_f_count}",
}
_PGC_FIT_UPDATE_STR: str = "{pgc_fitness} = " + _WEIGHTED_FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_PGC_F_COUNT_UPDATE_STR: str = "{pgc_f_count} = " + _FIXED_UPDATE_RAW.format_map(_PGC_FIT_UPDATE_MAP)
_EVO_UPDATE_MAP: dict[str, str] = {
    "CSCV": "EXCLUDED.{evolvability}",
    "CSCC": "EXCLUDED.{e_count}",
    "PSCV": '"__table__".{evolvability}',
    "PSCC": '"__table__".{e_count}',
    "CSPV": "EXCLUDED.{_evolvability}",
    "CSPC": "EXCLUDED.{_e_count}",
}
_EVO_UPDATE_STR: str = "{evolvability} = " + _WEIGHTED_SCALAR_UPDATE.format_map(_EVO_UPDATE_MAP)
_EVO_COUNT_UPDATE_STR: str = "{e_count} = " + _SCALAR_COUNT_UPDATE.format_map(_EVO_UPDATE_MAP)
_REF_UPDATE_MAP: dict[str, str] = {
    "CSCC": "EXCLUDED.{reference_count}",
    "PSCC": '"__table__".{reference_count}',
    "CSPC": "EXCLUDED.{_reference_count}",
}
_REF_UPDATE_STR: str = "{reference_count} = " + _SCALAR_COUNT_UPDATE.format_map(_REF_UPDATE_MAP)
UPDATE_STR: str = ",".join(
    (
        "{updated} = NOW()",
        _PGC_EVO_UPDATE_STR,
        _PGC_E_COUNT_UPDATE_STR,
        _PGC_FIT_UPDATE_STR,
        _PGC_F_COUNT_UPDATE_STR,
        _EVO_UPDATE_STR,
        _EVO_COUNT_UPDATE_STR,
        _REF_UPDATE_STR,
    )
)


# Tree structure
_LEL: Literal["gca"] = "gca"
_REL: Literal["gcb"] = "gcb"
_NL: Literal["signature"] = "signature"
_PEL: Literal["pgc"] = "pgc"
_PTR_MAP: PtrMap = {_LEL: _NL, _REL: _NL}
_PTR_MAP_PLUS_PGC: PtrMap = {_LEL: _NL, _REL: _NL, _PEL: _NL}


# Data schema
with open(join(dirname(__file__), "formats/gms_table_format.json"), "r", encoding="utf-8") as file_ptr:
    GMS_RAW_TABLE_SCHEMA: dict[str, Any] = load(file_ptr)


class genetic_material_store:
    """Base class for all genetic material stores."""

    def __init__(self, config: TableConfig | TableConfigNorm, data_files: list[str]) -> None:
        """Connect to or create a genomic library.

        The genomic library data persists in a postgresql database. Multiple
        instances of the genomic_library() class can connect to the same database
        (use the same configuration).

        Args
        ----
        config(pypgtable config): The config is deep copied by pypgtable.
        """
        super().__init__()
        self.library: table = table(config)
        self._update_str: str = UPDATE_STR.replace("__table__", config["table"])
        self.encode_value: Callable[..., Any] = self.library.encode_value
        self.select: Callable[..., RowIter] = self.library.select
        self.recursive_select: Callable[..., RowIter] = self.library.recursive_select
        if self.library.raw.creator:
            self._creator(data_files)

    def __getitem__(self, signature: memoryview) -> dict[str, Any]:
        """Pull a genetic code from the genomic library."""
        entry = tuple(self.select("WHERE {signature} = {sig}", {"sig": signature}))
        if entry:
            return entry[0]
        raise KeyError

    def _creator(self, data_files: list[str]) -> None:
        """Called if this instance created the library table in the database."""
        raise NotImplementedError("The creator method must be implemented by the subclass.")

    def get(self, signature: memoryview) -> tuple[dict[str, Any], ...]:
        """Recursively pull a genetic code from the genomic library."""
        entry = tuple(self.recursive_select("WHERE {signature} = {sig}", {"sig": signature}))
        if entry:
            return entry
        raise KeyError

    @lru_cache(maxsize=10000)
    def signature_exists(self, signature: memoryview) -> bool:
        """Check if a signature exists in the genomic library."""
        return bool(
            tuple(
                self.library.select(
                    "WHERE {signature} = {sig}",
                    {"sig": signature},
                    ("signature",),
                    "tuple",
                )
            )
        )
