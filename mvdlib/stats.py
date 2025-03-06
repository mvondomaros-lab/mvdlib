from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def acf(
    x: NDArray[np.float64],
    fft: bool = True,
) -> NDArray[np.float64]:
    """
    Compute the autocorrelation function (ACF) of a signal.

    :param x: The input signal.
    :param fft: Whether to use the FFT-based algorithm. Default is True.

    :return: The ACF of the input signal.
    """
    import statsmodels.tsa.api

    nlags = x.size - 1
    res = statsmodels.tsa.api.acf(x - np.mean(x), nlags=nlags, adjusted=True, fft=fft)
    return res
