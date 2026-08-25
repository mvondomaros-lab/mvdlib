from math import ceil

import numba
import numpy as np
from numpy.typing import NDArray


@numba.vectorize(
    [numba.float64(numba.complex128)],
    nopython=True,
    fastmath=True,
)
def cabs2(
    x: complex | NDArray[np.complex128],
) -> float | NDArray[np.float64]:
    """
    Return the squared magnitude of a complex number.
    """
    return x.real * x.real + x.imag * x.imag


def prevpow2(x: int | float) -> int:
    """
    Return the largest power of two not exceeding ``abs(x)``, clamped to one.
    """
    n = int(abs(x))
    return 1 if n == 0 else 1 << n.bit_length() - 1


def nextpow2(x: int | float) -> int:
    """
    Return the smallest power of two not less than ``abs(x)``, clamped to one.
    """
    n = ceil(abs(x))
    return 1 if n == 0 else 1 << (n - 1).bit_length()
