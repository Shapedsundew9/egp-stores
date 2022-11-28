"""Common routines."""

from datetime import datetime, timedelta, timezone
from random import choice, getrandbits

EST = timezone(timedelta(hours=-5))
EGP_EPOCH = datetime(2019, 12, 25, 16, 26, tzinfo=EST)
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
