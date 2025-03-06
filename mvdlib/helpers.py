from __future__ import annotations

import typing

import numba
import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


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


@numba.jit(nopython=True, fastmath=True)
def _laplace_core(
    x: NDArray[np.float64], t: NDArray, s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the numerical Laplace transform of a signal.

    :param x: Input signal.
    :param t: Time values.
    :param s: Laplace frequency.

    :return: The numerical Laplace transform.
    """
    res = np.empty_like(s)
    for i in numba.prange(s.size):
        res[i] = np.trapezoid(x * np.exp(-s[i] * t), t)
    return res


def laplace(
    x: NDArray[np.float64], t: NDArray, s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the numerical Laplace transform of a signal using a Riemann sum.

    :param x: Input signal.
    :param t: Time values.
    :param s: Laplace frequency.

    :return: The numerical Laplace transform.
    """
    if x.ndim != 1:
        raise ValueError("Input signal must be 1-dimensional.")
    if t.ndim != 1:
        raise ValueError("Time values must be 1-dimensional.")
    if s.ndim != 1:
        raise ValueError("Laplace frequencies must be 1-dimensional.")
    if x.size < 2:
        raise ValueError("Input signal must have at least 2 points.")
    if x.size != t.size:
        raise ValueError("Input signal and time values must have the same size.")
    if not np.all(np.isclose(np.diff(t), t[1] - t[0])):
        raise ValueError("Time values must be evenly spaced.")
    return _laplace_core(x, t, s)
