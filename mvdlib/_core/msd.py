import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True, fastmath=True)
def msd_naive(x: NDArray[np.float64], maxsteps: int) -> NDArray[np.float64]:
    """Direct MSD."""
    out = np.empty(maxsteps, dtype=np.float64)
    for lag in range(maxsteps):
        n = x.size - lag
        acc = 0.0
        for i in range(n):
            dx = x[i + lag] - x[i]
            acc += dx * dx
        out[lag] = acc / n
    return out



@numba.njit(fastmath=True)
def mssq(x: NDArray[np.float64], maxsteps: int) -> NDArray[np.float64]:
    """
    Compute the time-lagged mean sum of squares of a trajectory.
    """
    xsq = x**2
    ssq = np.zeros(maxsteps, dtype=np.float64)
    ssq[0] = 2.0 * np.sum(xsq)
    for n in range(1, maxsteps):
        ssq[n] = ssq[n - 1] - xsq[n - 1] - xsq[-n]
    norm = np.arange(x.size, x.size - maxsteps, -1)
    mssq = ssq / norm
    return mssq


@numba.njit(fastmath=True)
def unwrap(x: NDArray[np.float64], box: float) -> NDArray[np.float64]:
    """
    Unwrap a one-dimensional periodic trajectory.
    """
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        dx = x[i] - y[i - 1]
        y[i] = x[i] - np.round(dx / box) * box
    return y
