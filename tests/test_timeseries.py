import numpy as np
import pytest
from numpy.typing import NDArray

from mvdlib import timeseries


@pytest.mark.parametrize(
    "x, nc, expected",
    [
        (np.zeros(10), None, np.zeros(10)),
        (np.ones(10), None, np.arange(10, 0, -1)),
        (np.zeros(10), 5, np.zeros(5)),
        (np.ones(10), 5, np.arange(10, 5, -1)),
    ],
)
def test_acf_naive_basic(
    x: NDArray[np.float64], nc: int | None, expected: NDArray[np.float64]
) -> None:
    acf = timeseries.acf_naive(x, nc)
    np.testing.assert_allclose(acf, expected)


def test_acf_naive_exponential() -> None:
    nx = 1000
    nc = 100
    x = np.exp(-np.arange(nx))
    acf = timeseries.acf_naive(x, nc)
    expected = x[:nc] / (1.0 - np.exp(-2.0))
    np.testing.assert_allclose(acf, expected)


@pytest.mark.parametrize(
    "x, nc, expected",
    [
        (np.zeros(10), None, np.zeros(10)),
        (np.ones(10), None, np.arange(10, 0, -1)),
        (np.zeros(10), 5, np.zeros(5)),
        (np.ones(10), 5, np.arange(10, 5, -1)),
    ],
)
def test_acf_basic(
    x: NDArray[np.float64], nc: int | None, expected: NDArray[np.float64]
) -> None:
    acf = timeseries.acf(x, nc)
    np.testing.assert_allclose(acf, expected)


def test_acf_exponential() -> None:
    nx = 1000
    nc = 100
    x = np.exp(-np.arange(nx))
    acf = timeseries.acf(x, nc)
    expected = x[:nc] / (1.0 - np.exp(-2.0))
    np.testing.assert_allclose(acf, expected, atol=1e-14)


@pytest.mark.parametrize("nx, nc", [(8, 1), (8, 4), (16, 8), (127, 32), (999, 123)])
def test_acf_matches_naive_implementation(nx: int, nc: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(nx)

    acf = timeseries.acf(x, nc)
    acf_naive = timeseries.acf_naive(x, nc)

    np.testing.assert_allclose(acf, acf_naive)


@pytest.mark.parametrize("func", [timeseries.acf, timeseries.acf_naive, timeseries.tcf])
def test_correlation_rejects_invalid_nc(func) -> None:
    x = np.arange(5, dtype=np.float64)

    with pytest.raises(TypeError, match="nc must be an integer"):
        func(x, nc=2.0)

    with pytest.raises(ValueError, match="nc must be positive"):
        func(x, nc=0)


def test_tcf_rejects_short_signal() -> None:
    with pytest.raises(ValueError, match="x must not be empty"):
        timeseries.tcf(np.array([], dtype=np.float64))

    with pytest.raises(ValueError, match="x must contain at least two points"):
        timeseries.tcf(np.array([0.0]))


@pytest.mark.parametrize(
    "x, shift, scale, expected",
    [
        (np.zeros(10), True, False, np.zeros(10)),
        (np.ones(10), True, False, np.zeros(10)),
        (np.ones(10), False, False, np.ones(10)),
    ],
)
def test_tcf_basic(
    x: NDArray[np.float64],
    shift: bool,
    scale: bool,
    expected: NDArray[np.float64],
) -> None:
    tcf = timeseries.tcf(x, shift=shift, scale=scale)
    np.testing.assert_allclose(tcf, expected)


def test_tcf_zero_lag() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(1.0 + 10.0 * rng.standard_normal(100))
    tcf = timeseries.tcf(x, shift=True, scale=True)
    assert tcf[0] == pytest.approx(1.0)


def test_tcf_exponential() -> None:
    nx = 1000
    nc = 100
    x = np.exp(-np.arange(nx))
    tcf = timeseries.tcf(x, nc=nc, shift=False)
    expected = (
        np.exp(-np.arange(nc)) / np.arange(nx, nx - nc, -1) / (1.0 - np.exp(-2.0))
    )
    np.testing.assert_allclose(tcf, expected, atol=1e-14)


def test_tcf_large_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000)
    tcf = timeseries.tcf(x)
    assert tcf.shape == x.shape
