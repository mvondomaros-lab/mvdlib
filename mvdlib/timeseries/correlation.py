import numbers

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft

# noinspection PyProtectedMember
from mvdlib._core.math import cabs2, nextpow2


def acf(x: NDArray[np.float64], nc: int | None = None) -> NDArray[np.float64]:
    """
    Compute the autocorrelation function of a signal using the FFT.

    Parameters
    ----------
    x
        Input signal.
    nc
        Number of correlation coefficients to compute. Defaults to ``x.size``.

    Returns
    -------
    numpy.ndarray
        Autocorrelation function.
    """
    x = _validate_signal(x)
    nx = x.size
    nc = _validate_nc(nc, nx)

    nf = nextpow2(nx + nc - 1)

    f = rfft(x, nf)
    s = cabs2(f)
    c = irfft(s)

    return c[:nc]


def tcf(
    x: NDArray[np.float64],
    nc: int | None = None,
    *,
    shift: bool = True,
    scale: bool = False,
) -> NDArray[np.float64]:
    """
    Compute the time correlation function of a signal.

    The time correlation function is the autocorrelation function normalized
    by the number of samples contributing to each lag.

    Parameters
    ----------
    x
        Input signal.
    nc
        Number of correlation coefficients to compute. Defaults to ``x.size``.
    shift
        Shift the signal to zero mean before computing the correlation.
    scale
        Scale the signal to unit standard deviation before computing the
        correlation.

    Returns
    -------
    numpy.ndarray
        Time correlation function.
    """
    x = _validate_signal(x)
    nc = _validate_nc(nc, x.size)
    x = _process_signal(x, shift, scale)
    c = acf(x, nc)
    lags = np.arange(x.size, x.size - c.size, -1)
    c /= lags
    return c


@numba.njit(fastmath=True)
def acf_naive(x: NDArray[np.float64], nc: int | None = None) -> NDArray[np.float64]:
    """
    Compute the autocorrelation function of a signal by direct summation.

    Parameters
    ----------
    x
        Input signal.
    nc
        Number of correlation coefficients to compute. Defaults to ``x.size``.

    Returns
    -------
    numpy.ndarray
        Autocorrelation function.
    """
    x = _validate_signal(x)
    nx = x.size
    nc = _validate_nc(nc, nx)
    c = np.zeros(nc, dtype=np.float64)

    for j in range(nc):
        acc = 0.0
        for i in range(nx - j):
            acc += x[i] * x[i + j]
        c[j] = acc
    return c


def _process_signal(
    x: NDArray[np.float64],
    shift: bool,
    scale: bool,
) -> NDArray[np.float64]:
    """
    Optionally shift and scale a signal.
    """
    if shift:
        x = x - np.mean(x)
    if scale:
        std = np.std(x)
        if std > 0.0:
            x = x / std
    return x


def _validate_signal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Validate a one-dimensional signal.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size == 0:
        raise ValueError("x must not be empty")
    return x


def _validate_nc(nc: int | None, nx: int) -> int:
    """
    Validate the number of correlation coefficients.
    """
    if nc is None:
        return nx
    if not isinstance(nc, numbers.Integral):
        raise TypeError("nc must be an integer")
    if nc <= 0:
        raise ValueError("nc must be positive")
    return min(int(nc), nx)
