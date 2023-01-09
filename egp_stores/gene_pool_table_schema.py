"""Common dependency of GP & GPC"""
from os.path import dirname, join
from json import load
from copy import deepcopy
from egp_types.gc_type_tools import merge
from .genetic_material_store import GMS_TABLE_SCHEMA

GP_TABLE_SCHEMA = deepcopy(GMS_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gp_table_format.json"), "r") as file_ptr:
    merge(GP_TABLE_SCHEMA, load(file_ptr))
