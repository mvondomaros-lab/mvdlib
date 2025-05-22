from __future__ import annotations

import typing

import numba
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Union

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
    Calculate the squared magnitude of a complex number or an array of complex numbers.

    :param x: Complex number(s).
    :return: The squared magnitude of the complex number(s).
    """
    return x.real * x.real + x.imag * x.imag


def prevpow2(x: Union[int, float]) -> int:
    """
    Calculate the largest power of 2 smaller than or equal to the absolute value
    of the given input.

    :param x: The input number.
    :return: The largest power of 2.
    """
    n = int(abs(x))
    return 1 if n == 0 else 1 << n.bit_length() - 1


def nextpow2(x: Union[int, float]) -> int:
    """
    Calculate the smallest power of 2 greater than or equal to the absolute value
    of the given input.

    :param x: The input number.
    :return: The next power of 2.
    """
    n = int(abs(x))
    return 1 if n == 0 else 1 << (n - 1).bit_length()


@numba.jit(nopython=True, fastmath=True)
def unwrap(x: NDArray[np.float64], box: float) -> None:
    """
    Remove jumps across periodic boundaries between consecutive elements of the input array.

    Operates in place.

    :param x: The input array.
    :param box: The periodic box length.
    """
    for i in range(1, x.size):
        dx = x[i] - x[i - 1]
        x[i] -= np.round(dx / box) * box
