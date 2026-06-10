import numbers
from collections.abc import Callable

import numba
import numpy as np
from numpy.typing import NDArray

from mvdlib._core import ld as _diffusion
from mvdlib.types import ScalarFloatFunc


def ld(
    *,
    friction: ScalarFloatFunc | float,
    nsteps: int,
    dt: float,
    mass: float,
    kt: float,
    x0: float,
    v0: float | None = None,
    force: ScalarFloatFunc | None = None,
    save_freq: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate one-dimensional underdamped Langevin dynamics.

    The dynamics are governed by

        m dv = force(x) dt - friction(x) v dt + sqrt(2 kT friction(x)) dW,
        dx = v dt.

    The corresponding local Smoluchowski diffusion coefficient is
    ``D(x) = kt / friction(x)``.

    Parameters
    ----------
    friction
        Friction coefficient. Either a scalar or a function of position.
    nsteps
        Number of integration steps.
    dt
        Integration timestep.
    mass
        Particle mass.
    kt
        Thermal energy ``kT``.
    x0
        Initial position.
    v0
        Initial velocity. Defaults to ``sqrt(kt / mass)``.
    force
        Position-dependent force. Defaults to zero.
    save_freq
        Save trajectory every ``save_freq`` integration steps. The initial
        state is always saved. ``nsteps`` must be divisible by ``save_freq``.
    rng
        NumPy random number generator. Defaults to
        ``np.random.default_rng()``.

    Returns
    -------
    tuple of numpy.ndarray
        Saved positions and velocities. Both arrays have length
        ``nsteps // save_freq + 1``.

    Raises
    ------
    TypeError
        If an argument has an incompatible type.
    ValueError
        If a numeric argument is non-finite, non-positive where positivity is
        required, or if ``nsteps`` is not divisible by ``save_freq``.
    """
    if not isinstance(nsteps, numbers.Integral):
        raise TypeError("nsteps must be an integer")
    if nsteps <= 0:
        raise ValueError("nsteps must be positive")

    if not isinstance(save_freq, numbers.Integral):
        raise TypeError("save_freq must be an integer")
    if save_freq <= 0:
        raise ValueError("save_freq must be positive")
    if nsteps % save_freq != 0:
        raise ValueError("nsteps must be divisible by save_freq")

    dt = _validate_positive_float("dt", dt)
    mass = _validate_positive_float("mass", mass)
    kt = _validate_positive_float("kt", kt)
    x0 = _validate_finite_float("x0", x0)

    if v0 is None:
        v0 = np.sqrt(kt / mass)
    else:
        v0 = _validate_finite_float("v0", v0)

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    if force is None:
        force_func = _constant_scalar_function(0.0)
    elif callable(force):
        force_func = _jit_scalar_function("force", force)
    else:
        raise TypeError("force must be callable or None")

    if isinstance(friction, numbers.Real):
        friction = _validate_positive_float("friction", friction)
        friction_func = _constant_scalar_function(friction)
    elif callable(friction):
        friction_func = _jit_scalar_function("friction", friction)
    else:
        raise TypeError("friction must be a real number or callable")

    force_value = force_func(x0)
    if not isinstance(force_value, numbers.Real):
        raise TypeError("force(x0) must return a real number")
    if not np.isfinite(force_value):
        raise ValueError("force(x0) must be finite")

    friction_value = friction_func(x0)
    if not isinstance(friction_value, numbers.Real):
        raise TypeError("friction(x0) must return a real number")
    if not np.isfinite(friction_value):
        raise ValueError("friction(x0) must be finite")
    if friction_value <= 0.0:
        raise ValueError("friction(x0) must be positive")

    return _diffusion.ld(
        force=force_func,
        friction=friction_func,
        nsteps=int(nsteps),
        dt=dt,
        mass=mass,
        kt=kt,
        x0=x0,
        v0=v0,
        save_freq=int(save_freq),
        rng=rng,
    )


def _constant_scalar_function(value: float) -> Callable[[float], float]:
    @numba.njit(fastmath=True)
    def func(_: float) -> float:
        return value

    return func


def _validate_finite_float(name: str, value: float) -> float:
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    value = _validate_finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _jit_scalar_function(
    name: str,
    func: Callable[[float], float],
) -> Callable[[float], float]:
    if isinstance(func, numba.core.registry.CPUDispatcher):
        return func

    try:
        return numba.njit(func)
    except Exception as error:
        raise TypeError(f"{name} must be Numba-compatible") from error
