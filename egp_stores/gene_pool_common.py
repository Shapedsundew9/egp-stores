"""Common dependencies of the GP"""
from copy import deepcopy
from json import load
from os.path import dirname, join
from typing import Any

from egp_utils.common import merge

from .genetic_material_store import GMS_RAW_TABLE_SCHEMA

GP_RAW_TABLE_SCHEMA: dict[str, Any] = deepcopy(GMS_RAW_TABLE_SCHEMA)
with open(
    join(dirname(__file__), "formats/gp_table_format.json"), "r", encoding="utf-8"
) as file_ptr:
    merge(GP_RAW_TABLE_SCHEMA, load(file_ptr))
GP_HIGHER_LAYER_COLS: tuple[str, ...] = tuple(
    (key for key in filter(lambda x: x[0] == "_", GP_RAW_TABLE_SCHEMA))
)
GP_UPDATE_RETURNING_COLS: tuple[str, ...] = tuple(
    (x[1:] for x in GP_HIGHER_LAYER_COLS)
) + ("updated", "created")
GP_REFERENCE_COLUMNS: tuple[str, ...] = tuple(
    k
    for k, _ in filter(
        lambda x: x[1].get("reference", False), GP_RAW_TABLE_SCHEMA.items()
    )
)
