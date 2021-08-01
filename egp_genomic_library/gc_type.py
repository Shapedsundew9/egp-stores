"""Manages genetic code type interactions.

Genetic code types are identified by a signed 16-bit value.
"""


from json import load
from os.path import dirname, join

# Load type data
# TODO: Could convert numerical dict to a list if they are contiguous and start at 0.
with open(join(dirname(__file__), "data/gc_types.json"), "r") as file_ptr:
    gc_type_lookup = load(file_ptr)
    gc_type_lookup['v2n'] = {int(k): v for k, v in gc_type_lookup['v2n'].items()}
    gc_type_lookup['n2v'] = {k: int(v) for k, v in gc_type_lookup['n2v'].items()}


def validate(gc_type_int):
    """Validate a gc_type_int.

    Args
    ----
    gc_type_int (int): The GC type value to validate.

    Returns
    -------
    (bool) True if the type is defined else false.
    """
    return gc_type_int in gc_type_lookup['v2n']


def asint(gc_type_str):
    """Convert a gc_type_str to its value representation.

    Args
    ----
    gc_type_str (str): The GC type name to convert to a value.

    Returns
    -------
    (int) GC type value.
    """
    return gc_type_lookup['n2v'].get(gc_type_str, gc_type_lookup['v2n']['egp.invalid_type'])


def asstr(gc_type_int):
    """Convert a gc_type_int to its string representation (name).

    Args
    ----
    gc_type_int (int): The GC type name to convert to a string.

    Returns
    -------
    (str) GC type name.
    """
    return gc_type_lookup['v2n'].get(gc_type_int, gc_type_lookup['n2v']['egp.invalid_type'])
