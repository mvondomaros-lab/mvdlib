import numpy as np
import pytest
from numpy.typing import NDArray

from mvdlib import diffusion


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.zeros(5), np.zeros(5)),
        (np.ones(5), np.zeros(5)),
        (np.arange(5), np.arange(5) ** 2),
    ],
)
def test_msd(x: NDArray[np.float64], expected: NDArray[np.float64]) -> None:
    msd = diffusion.msd(x)
    np.testing.assert_allclose(msd, expected)


def test_msd_maxsteps() -> None:
    x = np.arange(5, dtype=np.float64)
    msd = diffusion.msd(x, maxsteps=2)
    np.testing.assert_allclose(msd, np.array([0.0, 1.0]))


def test_msd_rejects_invalid_maxsteps() -> None:
    with pytest.raises(TypeError, match="maxsteps must be an integer"):
        # noinspection PyTypeChecker
        diffusion.msd(np.arange(5), maxsteps=2.5)

    with pytest.raises(ValueError, match="maxsteps must be positive"):
        diffusion.msd(np.arange(5), maxsteps=0)

    with pytest.raises(ValueError, match="maxsteps must not exceed x.size"):
        diffusion.msd(np.arange(5), maxsteps=6)


@pytest.mark.parametrize(
    "x, box, expected",
    [
        (np.array([1.0, 1.1]), 2.0, np.array([1.0, 1.1])),
        (np.array([1.0, 0.9]), 2.0, np.array([1.0, 0.9])),
        (np.array([0.0, 1.0]), 2.0, np.array([0.0, 1.0])),
        (np.array([0.0, 1.9]), 2.0, np.array([0.0, -0.1])),
        (np.array([0.0, 1.1]), 2.0, np.array([0.0, -0.9])),
        (np.array([1.9, 0.0]), 2.0, np.array([1.9, 2.0])),
        (np.array([1.1, 0.0]), 2.0, np.array([1.1, 2.0])),
    ],
)
def test_msd_unwraps_periodic_trajectory(
    x: NDArray[np.float64], box: float, expected: NDArray[np.float64]
) -> None:
    msd = diffusion.msd(x, box=box)
    expected_msd = diffusion.msd(expected)
    np.testing.assert_allclose(msd, expected_msd)


def test_ld_is_exposed() -> None:
    assert callable(diffusion.ld)


def test_ld_saves_initial_state() -> None:
    x, v = diffusion.ld(
        friction=1.0,
        nsteps=10,
        dt=0.01,
        mass=1.0,
        kt=1.0,
        x0=2.0,
        v0=0.0,
        save_freq=2,
        rng=np.random.default_rng(0),
    )

    assert x.shape == (6,)
    assert v.shape == (6,)
    assert x[0] == 2.0
    assert v[0] == 0.0


def test_ld_is_reproducible_with_seeded_rng() -> None:
    kwargs = dict(
        friction=1.0,
        nsteps=10,
        dt=0.01,
        mass=1.0,
        kt=1.0,
        x0=0.0,
        v0=0.0,
        save_freq=1,
    )

    x1, v1 = diffusion.ld(**kwargs, rng=np.random.default_rng(1))
    x2, v2 = diffusion.ld(**kwargs, rng=np.random.default_rng(1))

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(v1, v2)


def test_ld_scalar_and_callable_friction_agree() -> None:
    def friction(_: float) -> float:
        return 1.0

    kwargs = dict(
        nsteps=10,
        dt=0.01,
        mass=1.0,
        kt=1.0,
        x0=0.0,
        v0=0.0,
        save_freq=1,
    )

    x1, v1 = diffusion.ld(friction=1.0, **kwargs, rng=np.random.default_rng(2))
    x2, v2 = diffusion.ld(friction=friction, **kwargs, rng=np.random.default_rng(2))

    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"nsteps": 0}, ValueError),
        ({"save_freq": 0}, ValueError),
        ({"nsteps": 10, "save_freq": 3}, ValueError),
        ({"dt": 0.0}, ValueError),
        ({"mass": 0.0}, ValueError),
        ({"kt": 0.0}, ValueError),
        ({"friction": 0.0}, ValueError),
        ({"rng": object()}, TypeError),
    ],
)
def test_ld_rejects_invalid_inputs(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    params = {
        "friction": 1.0,
        "nsteps": 10,
        "dt": 0.01,
        "mass": 1.0,
        "kt": 1.0,
        "x0": 0.0,
        "v0": 0.0,
        "save_freq": 1,
        "rng": np.random.default_rng(0),
        **kwargs,
    }

    with pytest.raises(error):
        diffusion.ld(**params)
