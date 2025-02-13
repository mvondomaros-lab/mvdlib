import random
from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from mvdlib import stats


@pytest.mark.parametrize(
    "x, nc, expected",
    [
        (np.zeros(10), None, np.zeros(10)),  # ACF of zeros should be zero.
        (np.ones(10), None, np.arange(10, 0, -1)),  # ACF of ones should be triangular.
        (np.zeros(10), 5, np.zeros(5)),
        (np.ones(10), 5, np.arange(10, 5, -1)),
    ],
)
def test_acf_naive_basic(x: NDArray, nc: Optional[int], expected: NDArray) -> None:
    """Test basic cases for the naive ACF implementation."""
    assert stats.acf_naive(x, nc) == pytest.approx(expected)


def test_acf_naive_exponential() -> None:
    """Test ACF of an exponential signal."""
    nx = 1000
    nc = 100
    x = np.exp(-np.arange(nx))
    acf = stats.acf_naive(x, nc)
    assert acf == pytest.approx(x[:nc] / (1.0 - np.exp(-2.0)))


@pytest.mark.parametrize(
    "x, nc, expected",
    [
        (np.zeros(10), None, np.zeros(10)),  # ACF of zeros should be zero.
        (np.ones(10), None, np.arange(10, 0, -1)),  # ACF of ones should be triangular.
        (np.zeros(10), 5, np.zeros(5)),
        (np.ones(10), 5, np.arange(10, 5, -1)),
    ],
)
def test_acf_basic(x: NDArray, nc: Optional[int], expected: NDArray) -> None:
    """Test basic cases for the FFT-based ACF implementation."""
    assert stats.acf(x, nc) == pytest.approx(expected)


def test_acf_exponential() -> None:
    """Test ACF of an exponential signal."""
    nx = 1000
    nc = 100
    x = np.exp(-np.arange(nx))
    acf = stats.acf(x, nc)
    assert acf == pytest.approx(x[:nc] / (1.0 - np.exp(-2.0)))


@pytest.mark.parametrize(
    "nx, nc", [(nx, random.randint(1, nx)) for nx in np.random.randint(2, 1000, 10)]
)
def test_acf_random_comparison(nx: int, nc: int) -> None:
    """Compare FFT-based ACF with naive ACF for random signals."""
    x = np.random.randn(nx)
    c = stats.acf(x, nc)
    c_naive = stats.acf_naive(x, nc)
    assert c == pytest.approx(c_naive)


@pytest.mark.parametrize(
    "x, shift, scale, expected",
    [
        (np.zeros(10), True, False, np.zeros(10)),  # TCF of zeros should be zero.
        (
            np.ones(10),
            True,
            False,
            np.zeros(10),
        ),  # TCF of ones should be zero if shifted.
        (
            np.ones(10),
            False,
            False,
            np.ones(10),
        ),  # TCF of ones should be one if not shifted.
    ],
)
def test_tcf_basic(x: NDArray, shift: bool, scale: bool, expected: NDArray) -> None:
    """Test basic cases for the TCF implementation."""
    assert stats.tcf(x, shift=shift, scale=scale) == pytest.approx(expected)


def test_tcf_zero_lag() -> None:
    """Test that TCF at zero lag is one after shifting and scaling."""
    x = np.cumsum(1.0 + 10.0 * np.random.randn(100))
    c = stats.tcf(x, shift=True, scale=True)
    assert c[0] == pytest.approx(1.0)


def test_tcf_exponential() -> None:
    """Test TCF of an exponential signal."""
    nx = 1000
    nc = 100
    x = np.exp(-np.arange(nx))
    c = stats.tcf(x, nc=nc, shift=False)
    c_ref = np.exp(-np.arange(nc)) / np.arange(nx, nx - nc, -1) / (1.0 - np.exp(-2.0))
    assert c == pytest.approx(c_ref)


def test_tcf_large_signal() -> None:
    """Test TCF with a large signal to ensure performance."""
    x = np.random.randn(10000)
    c = stats.tcf(x)
    assert c.size > 0  # Ensure computation is performed.
