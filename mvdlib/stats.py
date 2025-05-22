from __future__ import annotations

import typing

import numba
import numpy as np

from .helpers import prevpow2

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

    # Direct summation for each lag j.
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
        std = np.std(x)
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


def sem(x: NDArray[np.float64]) -> np.float64:
    """Compute the standard error of the mean (SEM) of uncorrelated data."""
    n = x.size
    return np.std(x) / np.sqrt(n)


def sem_corr(x: NDArray[np.float64]) -> np.float64:
    """
    Compute the standard error of the mean (SEM) of correlated data.

    Use the automated blocking method proposed by Jonsson (https://doi.org/10.1103/PhysRevE.98.043304). This is an
    adaptation of their code.
    """
    # Truncate the data to a power of 2.
    n = prevpow2(x.size)
    d = int(np.log2(n))
    x = x[:n]

    mu = np.mean(x)
    s, gamma = np.zeros(d), np.zeros(d)
    for i in np.arange(0, d):
        # Estimate the autocovariance and variance of x.
        gamma[i] = sum((x[: n - 1] - mu) * (x[1:] - mu)) / n
        s[i] = np.var(x)
        # Perform a blocking transformation.
        x = 0.5 * (x[0::2] + x[1::2])
        n = x.size

    # Compute m.
    m = (np.cumsum(((gamma / s) ** 2 * 2 ** np.arange(1, d + 1)[::-1])[::-1]))[::-1]

    # Magic numbers from the paper.
    q = np.array(
        [
            6.634897,
            9.210340,
            11.344867,
            13.276704,
            15.086272,
            16.811894,
            18.475307,
            20.090235,
            21.665994,
            23.209251,
            24.724970,
            26.216967,
            27.688250,
            29.141238,
            30.577914,
            31.999927,
            33.408664,
            34.805306,
            36.190869,
            37.566235,
            38.932173,
            40.289360,
            41.638398,
            42.979820,
            44.314105,
            45.641683,
            46.962942,
            48.278236,
            49.587884,
            50.892181,
        ]
    )

    # Determine when we should have stopped blocking.
    k = 0
    while k < d and m[k] >= q[k]:
        k += 1
    if k >= d - 1:
        raise ValueError("need more data for SEM estimation")
    return np.sqrt(s[k] / 2 ** (d - k))
