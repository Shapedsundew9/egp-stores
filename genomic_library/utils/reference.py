"""Common routines for Erasmus GP"""

from datetime import datetime
from random import choice, getrandbits


EGP_EPOCH = datetime(1975, 8, 7)
EGP_EMPTY_TUPLE = tuple()
_SIGN = (1, -1)


def random_reference():
    """Fast way to get a unique (enough) reference."""
    return getrandbits(63) * choice(_SIGN)


def sequential_reference():
    """Generate infinite reference sequence."""
    i = 0
    while True:
        yield i
        i += 1
