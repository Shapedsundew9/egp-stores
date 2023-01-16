"""Test the Gene Pool."""
from logging import Logger, NullHandler, getLogger
from typing import Any, Iterable

from egp_population.typing import PopulationNorm
from egp_stores.gene_pool import default_config as gp_default_config
from egp_stores.gene_pool import gene_pool
from egp_types.ep_type import vtype
from egp_types.xGC import xGC
from numpy import arange, array, clip, float32, int32, isfinite, where
from numpy.random import randint
from numpy.typing import NDArray
from pypgtable.typing import TableConfig
from uuid import uuid4
from hashlib import sha256
from datetime import datetime

from egp_stores.genomic_library import default_config as gl_default_config
from egp_stores.genomic_library import genomic_library

# Logging
_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())

# Create a genomic library
_GL_CONFIG: dict[str, Any] = gl_default_config()
_GL_CONFIG['database']['dbname'] = 'test_db'
_GL_CONFIG['delete_table'] = True
_GL: genomic_library = genomic_library(_GL_CONFIG)


# Gene pool config
_GP_CONFIG: dict[str, TableConfig] = gp_default_config()
for table in _GP_CONFIG:
    _GP_CONFIG[table]['delete_table'] = True
    _GP_CONFIG[table].setdefault('database', {})['dbname'] = 'test_db'


# Divide! Characterization function
_X1_TEST: NDArray[float32] = randint(-100, 100, (100,), dtype=int32).astype(float32)
_X2_TEST: NDArray[float32] = randint(-100, 100, (100,), dtype=int32).astype(float32)
_X2_TEST: NDArray[float32] = where(_X2_TEST == 0, arange(100, dtype=float32), _X2_TEST)
_Y_TEST: NDArray[float32] = _X1_TEST / _X2_TEST


def characterize_divide(gc: xGC) -> float:
    """"Characterize gc for Divide!

    Fitness is 1 - the normalised clipped mean-squared-error of 100 random division examples.
    The squared error is clipped to 10 and normalised to 1.

    Survivability is fitness in this test.

    NOTE: GGC may be modified in a limited way

    Args
    ----
    gc (gGC): The GC to be characterized.

    Returns
    -------
    tuple(float, float): Fitness, Survivability both in the range 0.0 <= x <= 1.0
    """
    results: list[float] = []
    for x1, x2, y in zip(_X1_TEST, _X2_TEST, _Y_TEST):
        result: tuple[float] | None = gc['exec']((x1, x2))
        if result is None:
            result = (100000.0,)
        results.append(result[0])
    y_pred: NDArray[float32] = array(results, dtype=float32)
    y_pred = where(isfinite(y_pred), y_pred, float32(100000.0))
    _logger.debug(f'GC {gc["ref"]} Predicted: {y_pred}')
    clipped: NDArray[float32] = clip((_Y_TEST - y_pred) ** 2, -10.0, 10.0)
    _logger.debug(f'GC {gc["ref"]} Clipped: {clipped}')
    mse: NDArray[float32] = clipped.mean() / float32(10.0)
    _logger.debug(f'GC {gc["ref"]} MSE: {mse}')
    fitness: float = (float32(1.0) - mse).sum()
    _logger.debug(f'GC {gc["ref"]} fitness = {fitness}, survivability = {fitness}')
    return fitness


def recharacterize_divide(gcs: Iterable[xGC]) -> tuple[float, ...]:
    """Re-characterize gc for Divide!

    The 'survivability'[0] value of each GC is modified.

    Args
    ----
    gcs (tuple(gGC)): The GCs to be recharacterized.
    """
    # In this case no modification is made.
    return tuple(gc['survivability'] for gc in gcs)


# A population config
_DIV_CONFIG: PopulationNorm = {
    'uid': 1,
    'worker_id': uuid4(),
    'fitness_function_hash': sha256(b'random string').digest(),
    'size': 100,
    'name': 'Divide!',
    'inputs': ('float', 'float'),
    'outputs': ('float',),
    'fitness_function': characterize_divide,
    'survivability_function': recharacterize_divide,
    'ordered_interface_hash': 123541235,
    'meta_data': None,
    'created': datetime.now(),
    'vt': vtype.EP_TYPE_STR,
    'description': 'Input values are x1, x2. Desired return value, y, is x1/x2.'
}

# Linear Fit Characterization function
_M: float = 17.42
_C: float = -4.2
_MSE_LIMIT: float = 2.0 ** 12
_MIN_FITNESS: float = 1.0 / _MSE_LIMIT


def characterize_fit(gc) -> tuple[float, float]:
    """"Characterize gc for a linear function fit!

    Fitness is means squared error of 101 with unit separation in the range
    -50 to +50

    Survivability is fitness in this test.

    NOTE: GGC may be modified in a limited way

    Args
    ----
    gc (gGC): The GC to be characterized.

    Returns
    -------
    tuple(float, float): Fitness, Survivability both in the range 0.0 <= x <= 1.0 or None
    """
    results: list[float] = [gc['exec']((x,)) for x in arange(-50, 51)]
    if None in results:
        fitness: float = 0.0
    else:
        mse: float = ((results - (arange(-50, 51) * _M + _C)) ** 2).mean()
        if not isfinite(mse):
            fitness = 0.0
        elif mse > _MSE_LIMIT:
            fitness = _MIN_FITNESS
        else:
            fitness = 1.0 - mse / _MSE_LIMIT
    _logger.debug(f'GC {gc["ref"]} fitness = {fitness}, survivability = {fitness}')
    return fitness, fitness


def recharacterize_fit(gcs) -> None:
    """Re-characterize gc for linear fit!

    The 'survivability'[0] value of each GC is modified.

    Args
    ----
    gcs (tuple(gGC)): The GCs to be recharacterized.
    """
    # In this case no modification is made.
    pass


# A population config
_FIT_CONFIG: dict[str, Any] = {
    'size': 100,
    'name': 'Linear fit',
    'inputs': ('float',),
    'outputs': ('float',),
    'characterize': characterize_fit,
    'recharacterize': recharacterize_fit,
    'vt': vtype.EP_TYPE_STR,
    'description': 'Input values are x. Desired return value, y = _M * x + _C'
}


def test_default_instanciation() -> None:
    """Simple instanciation."""
    gene_pool({_DIV_CONFIG['name']: _DIV_CONFIG}, uuid4(), _GL, _GP_CONFIG)
