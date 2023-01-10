"""Common dependencies of the GP"""
from os.path import dirname, join
from json import load
from copy import deepcopy
from egp_types.gc_type_tools import merge
from .genetic_material_store import GMS_TABLE_SCHEMA
from typing import Dict
from itertools import count


_REFERENCE_MASK = 0x7FFFFFFFFFFFFFFF
_GL_GC = 0x8000000000000000
_MAX_OWNER = 0x7FFFFFFF
GP_TABLE_SCHEMA = deepcopy(GMS_TABLE_SCHEMA)
with open(join(dirname(__file__), "formats/gp_table_format.json"), "r") as file_ptr:
    merge(GP_TABLE_SCHEMA, load(file_ptr))
GP_HIGHER_LAYER_COLS = tuple((key for key in filter(lambda x: x[0] == '_', GP_TABLE_SCHEMA)))
GP_UPDATE_RETURNING_COLS = tuple((x[1:] for x in GP_HIGHER_LAYER_COLS)) + ('updated', 'created')
GP_REFERENCE_COLUMNS = tuple((key for key, _ in filter(lambda x: x[1].get('reference', False), GP_TABLE_SCHEMA.items())))


# Pretty print for references
_OVER_MAX = 1 << 64
_MASK = _OVER_MAX - 1
ref_str = lambda x: 'None' if x is None else f"{((_OVER_MAX + x) & _MASK):016x}"


def ref_from_sig(signature:bytes, shift:int = 0) -> int:
    """Create a 63 bit reference from a signature.

    See reference() for significance of bit fields.
    shift can be used to make up to 193 alternate references
    
    Args
    ----
    signature: 32 element bytes object
    shift: Defines the lowest bit in the signature of the 63 bit reference

    Returns
    -------
    Reference    
    """
    if not shift:
        return int.from_bytes(signature[:8], "little") | _GL_GC 

    low = shift >> 3
    high = low + 9
    window = _REFERENCE_MASK << (shift & 0x3)
    return ((int.from_bytes(signature[low:high], "little") & window) >> shift) | _GL_GC
    

def reference(owner:int, counters:Dict[int, count]) -> int:
    """Create a unique reference.

    References have the structure:

    | Bit Field | Name | Description |
    ----------------------------------
    | 63 | GL | 0: Not in the GL, 1: In the GL |
    | 62:0 | TS | When GL = 1: TS = signature[62:0] |
    | 62:32 | OW | When GL = 0: Owner UID |
    | 31:0 | IX | When GL = 0: UID in the owner scope |

    Args
    ----
    owner: 32 bit unsigned integer uniquely identifying the counter to be used.

    Returns
    -------
    Signed 64 bit integer reference.
    """
    if owner not in counters:
        assert owner < _MAX_OWNER, "Owner index out of range."
        counters[owner] = count(2**32)
    return (counters[owner] + (owner << 32))
