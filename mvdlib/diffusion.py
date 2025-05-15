from __future__ import annotations

import typing

import numba
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Callable, Optional, Tuple

    from numpy.typing import NDArray


@numba.jit(nopython=True, fastmath=True)
def _ld_core(
    force: numba.types.Callable,
    friction: numba.types.Callable,
    nsteps: int,
    dt: float,
    mass: float,
    kt: float,
    x0: float,
    v0: float,
    save_freq: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate one-dimensional Langevin dynamics with a position-dependent external force and friction.

    Numba-optimized core function implementing the Euler--Maruyama integrator.

    :param force: The force exerted by the external potential.
    :param friction: The friction constant.
    :param nsteps: The number of simulation steps.
    :param dt: The time step.
    :param mass: The mass of the particle.
    :param kt: The thermal energy.
    :param x0: The initial position.
    :param v0: The initial velocity.
    :param save_freq: The trajectory save frequency.
    :param rng: A random number generator.

    :return: The simulated positions and velocities.
    """
    # Preallocate the position and velocity arrays.
    ns = nsteps // save_freq
    xs = np.empty(ns, dtype=np.float64)
    vs = np.empty(ns, dtype=np.float64)

    # No need to simulate more than we save.
    nsteps = ns * save_freq

    # Precompute the displacements due to the Wiener process.
    dtm = dt / mass
    noise = np.sqrt(2.0 * kt * dt) / mass * rng.standard_normal(size=nsteps)

    # Initialize position and velocity.
    x = x0
    v = v0

    # Ignore simulation steps that won't be saved by running only ns * save_freq steps.
    for i in range(nsteps):
        # Compute the position-dependent force and friction constant.
        f = force(x)
        zeta = friction(x)
        # Update velocity and position (Euler--Maruyama).
        v += f * dtm - zeta * v * dtm + np.sqrt(zeta) * noise[i]
        x += v * dt
        # Save with the requested frequency.
        if i % save_freq == 0:
            j = i // save_freq
            xs[j] = x
            vs[j] = v

    return xs, vs


def _ld_validate(nsteps: int, mass: float, diff: float, kt: float, dt: float):
    """
    Validate the parameters for simulating Langevin dynamics.

    Ensures that all input parameters meet the required bounds and stability conditions.

    :param nsteps: The number of simulation steps.
    :param mass: The mass of the particle; must be non-negative.
    :param diff: The diffusivity of the particle; must be non-negative.
    :param kt: The thermal energy (Boltzmann constant times temperature); must be non-negative.
    :param dt: The time step; must be non-negative and satisfy stability constraints.

    :raises ValueError: If any parameter is invalid or does not meet the required stability constraints.
    """
    # Check bounds.
    if nsteps < 0:
        raise ValueError("The number of simulation steps must be non-negative.")
    if mass < 0.0:
        raise ValueError("The mass must be non-negative.")
    if diff < 0.0:
        raise ValueError("The diffusivity must be non-negative.")
    if kt < 0.0:
        raise ValueError("The thermal energy must be non-negative.")
    if dt < 0.0:
        raise ValueError("The simulation time step must be non-negative.")

    # Check stability.
    if dt > 0.1 * diff * mass / kt:
        raise ValueError(
            "The simulation time step is too small. Use dt <= 0.1 * diff * mass / kt."
        )


def _constant_callable(value: float) -> Callable[[float], float]:
    """
    Create a constant callable that returns the same value for any input.

    :param value: The constant value to return.
    :return: A callable that returns the constant value.
    """

    @numba.jit(nopython=True, fastmath=True)
    def constant_callable(_: float) -> float:
        return value

    return constant_callable


def ld(
    *,
    friction: numba.types.Callable | float,
    nsteps: int,
    dt: float,
    mass: float,
    kt: float,
    x0: float,
    v0: Optional[float] = None,
    force: Optional[numba.types.Callable] = None,
    save_freq: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate one-dimensional Langevin dynamics with a position-dependent external force and friction.

    :param friction: A constant value or a function returning a position-dependent friction constant. Defaults to 1.0.
    :param nsteps: The number of simulation steps.
    :param dt: The time step.
    :param mass: The mass of the particle.
    :param kt: The thermal energy.
    :param x0: The initial position.
    :param v0: The initial velocity. Defaults to sqrt(kt / mass).
    :param force: A function returning a position-dependent external force. Defaults to `None`, meaning no force.
    :param save_freq: The trajectory save frequency. Defaults to 1 (save every step).
    :param rng: A random number generator. Defaults to `np.random.default_rng()`.

    :return: The simulated positions and velocities.
    """

    # Validate the simulation parameters.
    # _ld_validate(nsteps=nsteps, mass=mass, diff=diff, kt=kt, dt=dt)

    # Initialize the optional parameters.
    rng = rng or np.random.default_rng()

    force = force or _constant_callable(0.0)
    v0 = v0 or np.sqrt(kt / mass)

    # Initialize the friction parameter.
    friction = friction if callable(friction) else _constant_callable(friction)

    # Simulate Langevin dynamics in an external potential.
    x, v = _ld_core(
        force=force,
        friction=friction,
        nsteps=nsteps,
        dt=dt,
        mass=mass,
        kt=kt,
        x0=x0,
        v0=v0,
        save_freq=save_freq,
        rng=rng,
    )
    return x, v


@numba.jit(nopython=True, fastmath=True)
def _mssq(x: NDArray[np.float64], maxsteps: int) -> NDArray[np.float64]:
    """
    Calculate the time-lagged, mean sum of squares of the input trajectory.

    The mean sum of squares (MSSQ) is defined as
        MSSQ(n) = <x(n)^2 + x(0)^2>
                = SUM(x(n)^2 + x(0)^2) / (LENGTH(x) - n)
                = SSQ(n) / (LENGTH(x) - n)

    The sum of squares (SSQ) is computed efficiently using a recurrence relation,
        SSQ(0) = 2.0 * SUM(x^2)
        SSQ(1) = SSQ(0) - x(0)^2 - x(N-1)^2
        SSQ(2) = SSQ(1) - x(1)^2 - x(N-2)^2
        ...

    :param x: Input trajectory.
    :param maxsteps: Maximum number of MSSQ values to compute.
    :return: The MSSQ values.
    """
    xsq = x**2
    ssq = np.zeros(maxsteps, dtype=np.float64)
    ssq[0] = 2.0 * np.sum(xsq)
    for n in range(1, maxsteps):
        ssq[n] = ssq[n - 1] - xsq[n - 1] - xsq[-n]
    norm = np.arange(x.size, x.size - maxsteps, -1)
    mssq = ssq / norm
    return mssq


def msd(
    x: NDArray[np.float64],
    *,
    maxsteps: Optional[int] = None,
    box: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Compute the one-dimensional, mean-squared displacement (MSD) of the input trajectory.

    For an efficient computation, the mean squared displacement
        MSD(n) = <[x(n) - x(0)]^2>
    is separated into
        MSD(n) = MSSQ(n) - 2.0 * TCF(n),
    where
        TCF(n) = <x(n) * x(0)>
    is the time correlation function (TCF), and
        MSSQ(n) = <x(n)^2 + x(0)^2>
    is the time-lagged, mean sum of squares (MSSQ) of the input trajectory.

    The TCF is computed efficiently using the Fast Fourier Transform (FFT), and the mean sum of squares is computed
    using a recurrence relation.

    :param x: Input trajectory.
    :param maxsteps: Maximum number of MSD values to compute. If not provided, use the length of the input trajectory.
    :param box: Box size used for unwrapping. If provided, positions are unwrapped based on the given box dimensions.
    :return: The MSD values.
    """
    from .helpers import unwrap
    from .stats import tcf

    maxsteps = maxsteps or x.size

    if box is not None:
        unwrap(x, box)

    mssq = _mssq(x, maxsteps)
    tcf = tcf(x, maxsteps, shift=False)

    return mssq - 2.0 * tcf


@numba.jit(nopython=True, fastmath=True)
def interval_mask(x: NDArray[np.float64], xmin: float, xmax: float) -> NDArray[np.int8]:
    """
    Convert an array into a binary mask indicating which entries lie in [xmin, xmax).

    :param x: The input array.
    :param xmin: The lower bound.
    :param xmax: The upper bound.
    :returns: A binary mask where 1 indicates x[i] is in [xmin, xmax).
    """
    # Vectorized comparison yields a boolean array; cast to int8 in one go.
    return ((x >= xmin) & (x < xmax)).astype(np.int8)


def interval_survival_probability(
    x: NDArray[np.float64], xmin: float, xmax: float
) -> NDArray[np.float64]:
    """
    Compute the survival probability of a particle in an interval [xmin, xmax).

    The survival probability is defined as the probability to find the particle in the interval at time t, given that it
    was in the interval at time 0. By convention, S(0) = 1.

    :param x: The input array.
    :param xmin: The lower bound.
    :param xmax: The upper bound.
    :returns: The survival probabilities up to the maximum residence time.
    """
    # Compute binary mask indicating which entries lie in [xmin, xmax).
    mask = interval_mask(x, xmin, xmax)

    # Zero-pad both sides, so the ends are handled correctly.
    padded = np.pad(mask, (1, 1))

    # Compute run lengths (times spent in the interval).
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts

    if len(lengths) == 0:
        # No visits at all. Survival is zero for t ≥ 1, S(0) = 1 by convention.
        return np.array([1.0, 0.0])

    # Build a histogram of run lengths.
    counts = np.bincount(lengths)

    # Initialize the survival probability array and compute the normalization factor (i.e., the number of times that
    # the particle is within the interval).
    survival = np.zeros(counts.size, dtype=np.float64)
    norm = np.sum(np.arange(counts.size) * counts)

    # Count how often the particle is in the interval at time t, given that it was in the interval at time 0.
    for length, count in enumerate(counts):
        if count != 0:
            survival[:length] += np.arange(length, 0, -1) * count

    return survival / norm


def interval_residence_time(
    x: NDArray[np.float64], xmin: float, xmax: float, dt: float
) -> float:
    """
    Compute the mean residence time of a particle in an interval [xmin, xmax).

    The residence time is defined as integral over the survival probability.

    :param x: The input array.
    :param xmin: The lower bound.
    :param xmax: The upper bound.
    :param dt: The time step.
    :returns: The survival probabilities up to the maximum residence time.
    """
    survival = interval_survival_probability(x, xmin, xmax)
    residence = np.trapezoid(y=survival, dx=dt)
    # noinspection PyTypeChecker
    return residence


def interval_diffusivity(
    x: NDArray[np.float64], xmin: float, xmax: float, dt: float
) -> float:
    """
    Compute the mean diffusivity of a particle in an interval [xmin, xmax).

    The diffusivity is computed as

        D = (xmax - xmin)^2 / (12 * residence_time).

    This assumes a bulk environment, where the particle is free to diffuse in the interval.

    :param x: The input array.
    :param xmin: The lower bound.
    :param xmax: The upper bound.
    :param dt: The time step.
    :returns: The survival probabilities up to the maximum residence time.
    """
    residence = interval_residence_time(x, xmin, xmax, dt)
    return (xmax - xmin) ** 2 / (12.0 * residence)
