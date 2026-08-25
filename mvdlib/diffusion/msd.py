import numbers

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray

# noinspection PyProtectedMember
from mvdlib._core.msd import msd_naive as _msd_naive

# noinspection PyProtectedMember
from mvdlib._core.msd import mssq as _mssq

# noinspection PyProtectedMember
from mvdlib._core.msd import unwrap as _unwrap
from mvdlib.timeseries import tcf


def msd(
    x: ArrayLike,
    *,
    maxsteps: int | None = None,
    box: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute the one-dimensional mean-squared displacement of a trajectory.

    The mean-squared displacement is evaluated as

        MSD(n) = MSSQ(n) - 2 TCF(n),

    where ``MSSQ(n) = <x(n)^2 + x(0)^2>`` is the time-lagged mean sum of
    squares and ``TCF(n) = <x(n) x(0)>`` is the time correlation function.
    ``MSSQ`` is computed by a recurrence relation, while ``TCF`` is
    evaluated using an FFT-based correlation routine.

    Parameters
    ----------
    x
        Input trajectory.
    maxsteps
        Maximum number of time lags to compute. Defaults to ``x.size``.
    box
        Periodic box size used to unwrap the trajectory before computing the
        mean-squared displacement.

    Returns
    -------
    numpy.ndarray
        Mean-squared displacement values.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size < 2:
        raise ValueError("x must contain at least two points")

    if maxsteps is None:
        maxsteps = x.size
    elif not isinstance(maxsteps, numbers.Integral):
        raise TypeError("maxsteps must be an integer")
    elif maxsteps <= 0:
        raise ValueError("maxsteps must be positive")
    elif maxsteps > x.size:
        raise ValueError("maxsteps must not exceed x.size")
    maxsteps = int(maxsteps)

    if box is not None:
        if not isinstance(box, numbers.Real):
            raise TypeError("box must be a real number")
        box = float(box)
        if not np.isfinite(box):
            raise ValueError("box must be finite")
        if box <= 0.0:
            raise ValueError("box must be positive")
        x = _unwrap(x, box)

    x = x - x[0]
    return _mssq(x, maxsteps) - 2.0 * tcf(x, maxsteps, shift=False)


def msd_naive(
    x: ArrayLike,
    *,
    maxsteps: int | None = None,
    box: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute the one-dimensional mean-squared displacement by direct summation.

    Parameters
    ----------
    x
        Input trajectory.
    maxsteps
        Maximum number of time lags to compute. Defaults to ``x.size``.
    box
        Periodic box size used to unwrap the trajectory before computing the
        mean-squared displacement.

    Returns
    -------
    numpy.ndarray
        Mean-squared displacement values.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size < 2:
        raise ValueError("x must contain at least two points")

    if maxsteps is None:
        maxsteps = x.size
    elif not isinstance(maxsteps, numbers.Integral):
        raise TypeError("maxsteps must be an integer")
    elif maxsteps <= 0:
        raise ValueError("maxsteps must be positive")
    elif maxsteps > x.size:
        raise ValueError("maxsteps must not exceed x.size")
    maxsteps = int(maxsteps)

    if box is not None:
        if not isinstance(box, numbers.Real):
            raise TypeError("box must be a real number")
        box = float(box)
        if not np.isfinite(box):
            raise ValueError("box must be finite")
        if box <= 0.0:
            raise ValueError("box must be positive")
        x = _unwrap(x, box)

    x = x - x[0]
    return _msd_naive(x, maxsteps)
