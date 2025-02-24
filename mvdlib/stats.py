from __future__ import annotations

import typing

import numba
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray


def acf(x: NDArray[np.float64], nc: Optional[int] = None) -> NDArray[np.float64]:
    """
    Compute the autocorrelation function (ACF) of a signal.

    The computation is based on the Fast Fourier Transform (FFT).

    :param x: The input signal.
    :param nc: The number of correlation coefficients to compute. Defaults to the full length of the signal.
    :return: The ACF of the input signal.
    """
    from scipy.fft import irfft, rfft

    from .helpers import cabs2, nextpow2

    nx = x.size
    nc = min(int(nc or nx), nx)

    # Determine FFT length for zero-padding
    nf = nextpow2(nx + nc - 1)

    # Compute FFT, power spectrum, and inverse FFT
    f = rfft(x, nf)
    s = cabs2(f)
    c = irfft(s)

    return c[:nc]


@numba.jit(nopython=True, fastmath=True)
def acf_naive(x: NDArray[np.float64], nc: Optional[int] = None) -> NDArray[np.float64]:
    """
    Compute the autocorrelation function (ACF) of a signal.

    The computation is based on naive summation.

    :param x: The input signal.
    :param nc: The number of correlation coefficients to compute. Defaults to the full length of the signal.
    :return: The ACF of the input signal.
    """
    nx = x.size
    nc = min(int(nc or nx), nx)
    c = np.zeros(nc, dtype=np.float64)

    # Direct summation for each lag j
    for j in range(nc):
        acc = 0.0
        for i in range(nx - j):
            acc += x[i] * x[i + j]
        c[j] = acc
    return c


def _process_signal(
    x: NDArray[np.float64], shift: bool, scale: bool
) -> NDArray[np.float64]:
    """
    Shift and scale a signal by subtracting its mean and dividing by its standard deviation.

    :param x: The input signal.
    :param shift: Shift the signal to have zero mean?
    :param scale: Normalize the signal to have unit standard deviation?
    :return: The processed signal.
    """
    if shift:
        # This syntax creates a copy.
        x = x - np.mean(x)
    if scale:
        std = np.std(x, ddof=1)
        if std > 0.0:
            # This syntax creates a copy.
            x = x / std
    return x


def tcf(
    x: NDArray[np.float64],
    nc: Optional[int] = None,
    *,
    shift: Optional[bool] = True,
    scale: Optional[bool] = False,
) -> NDArray[np.float64]:
    """
    Compute the time correlation function (TCF).

    The TCF is an autocorrelation function (ACF) normalized by the decreasing number of samples contributing to the
    average for each lag. The computation is based on the Fast Fourier Transform (FFT).

    :param x: The input signal.
    :param nc: The number of correlation coefficients to compute. Defaults to the full length of the signal.
    :param shift: Shift the signal to have zero mean before processing? Defaults to True.
    :param scale: Normalize the signal to have unit standard deviation before processing? Defaults to False.
    :return: The TCF of the input signal.
    """
    x = _process_signal(x, shift, scale)
    c = acf(x, nc)
    lags = np.arange(x.size, x.size - c.size, -1)
    c /= lags
    return c
