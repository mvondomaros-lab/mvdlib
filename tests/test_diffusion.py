from typing import Optional

import numpy as np
import pytest
import scipy.stats
from numpy.typing import NDArray

from mvdlib import diffusion


@pytest.mark.parametrize(
    "mass, diff, kt, dt",
    [
        (1.0, 1.0, 1.0, 0.01),  # Default parameters.
        (1.0, 1.0, 1.0, 0.05),  # Different time step.
        (1.0, 2.0, 1.0, 0.01),  # Different mass.
        (2.0, 1.0, 1.0, 0.01),  # Different diffusivity.
        (1.0, 1.0, 2.0, 0.01),  # Different kt.
    ],
)
def test_ld(mass: float, diff: float, kt: float, dt: float) -> None:
    rng = np.random.default_rng(42)
    nsteps = 1000000
    x, v = diffusion.ld(nsteps, mass=mass, diff=diff, kt=kt, dt=dt, rng=rng)
    mean_v = np.mean(v)
    mean_v_expected = 0.0
    var_v = np.var(v)
    var_v_expected = kt / mass
    assert (x[0], v[0], mean_v, var_v) == (
        0.0,
        0.0,
        pytest.approx(mean_v_expected, abs=0.1),
        pytest.approx(var_v_expected, abs=0.1),
    )


@pytest.mark.parametrize(
    "nsteps, mass, diff, kt, dt",
    [
        (-1, 1.0, 1.0, 1.0, 0.1),  # Negative number of simulation steps.
        (1, -1.0, 1.0, 1.0, 0.1),  # Negative mass.
        (1, 1.0, -1.0, 1.0, 0.1),  # Negative diff.
        (1, 1.0, 1.0, -1.0, 0.1),  # Negative kt.
        (1, 1.0, 1.0, 1.0, -0.1),  # Negative dt.
        (1, 1.0, 1.0, 1.0, 1.0),  # dt too large.
    ],
)
def test_ld_validate(
    nsteps: int, mass: float, diff: float, kt: float, dt: float
) -> None:
    with pytest.raises(ValueError):
        diffusion.ld(nsteps, mass=mass, diff=diff, kt=kt, dt=dt)


@pytest.mark.parametrize(
    "n, D, dt, x0",
    [
        (100000, 0.1, 0.01, 0.0),  # Small D, small dt
        (100000, 0.3, 0.001, 1.0),  # Medium D, small dt
        (100000, 1.0, 0.1, -5.0),  # Large D, larger dt, negative x0
    ],
)
def test_random_walk_statistical_properties(
    n: int, D: float, dt: float, x0: float
) -> None:
    """Test that the random walk adheres to statistical properties."""
    rng = np.random.default_rng(42)
    x = diffusion.grw(n, diff=D, dt=dt, x0=x0, rng=rng)
    dx = np.diff(x)
    expected_std = np.sqrt(2.0 * D * dt)
    assert (x[0], np.mean(dx), np.std(dx), np.corrcoef(dx[:-1], dx[1:])[0, 1]) == (
        x0,
        pytest.approx(0.0, abs=1.0e-2),
        pytest.approx(expected_std, abs=1.0e-2),
        pytest.approx(0.0, abs=1.0e-2),
    )


@pytest.mark.parametrize(
    "x, nc, expected",
    [
        (np.zeros(5), None, np.zeros(5)),
        (np.ones(5), None, np.full(5, 2.0)),
        (np.arange(5), None, np.array([12.0, 11.0, 34.0 / 3.0, 13.0, 16.0])),
        (np.zeros(5), 2, np.zeros(2)),
        (np.ones(5), 2, np.full(2, 2.0)),
        (np.arange(5), 2, np.array([12.0, 11.0])),
    ],
)
def test_mssq(x: NDArray, nc: Optional[int], expected: NDArray) -> None:
    """Test the mean sum of squares function."""
    nc = nc or x.size
    assert diffusion._mssq(x, nc) == pytest.approx(expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.zeros(5), np.zeros(5)),
        (np.ones(5), np.zeros(5)),
        (np.arange(5), np.arange(5) ** 2),
    ],
)
def test_msd(x: NDArray, expected: NDArray) -> None:
    """Test the mean squared displacement function."""
    msd = diffusion.msd(x)
    assert msd == pytest.approx(expected)


def test_random_walk_msd() -> None:
    """Test the mean squared displacement for a random walk."""
    nx = 1000000
    nmax = 20
    x = diffusion.grw(nx, rng=np.random.default_rng(42))
    lags = np.arange(nmax)
    msds = diffusion.msd(x, maxsteps=nmax)
    slope, intercept, *_ = scipy.stats.linregress(lags, msds)
    assert (slope, intercept) == (
        pytest.approx(2.0, abs=1.0e-1),
        pytest.approx(0.0, abs=1.0e-1),
    )
