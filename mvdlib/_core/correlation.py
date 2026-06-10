import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(fastmath=True)
def acf_naive(x: NDArray[np.float64], nc: int) -> NDArray[np.float64]:
    """
    Compute the autocorrelation function by direct summation.
    """
    c = np.zeros(nc, dtype=np.float64)
    nx = x.size

    for j in range(nc):
        acc = 0.0
        for i in range(nx - j):
            acc += x[i] * x[i + j]
        c[j] = acc

    return c
