from __future__ import annotations

import typing

import numba
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Optional, Tuple

    from numpy.typing import NDArray


class USPACFAnalysis:
    def __init__(
        self,
        trajectories: NDArray[np.float64],
        dt: float = 1.0,
        c: float = 10.0,
    ):
        """
        Analyze the position autocorrelation functions of harmonically restrained MD simulations.

        :param trajectories: Input trajectories (positions).
        :param dt: The simulation time step.
        :param c: The constant c used in the windowing procedure.

        :raises ValueError: For erroneous trajectory inputs.
        """
        import mvdlib.stats

        # Compute the ACF.
        acfs = [mvdlib.stats.acf(x) for x in np.atleast_2d(trajectories)]
        self.acf = np.mean(acfs, axis=0)
        self.t = np.arange(self.acf.size) * dt

        # Compute the autocorrelation time.
        taus = np.cumsum(self.acf) - 0.5
        self.acfint = taus * dt
        cutoff = np.arange(self.acfint.size) <= c * np.abs(taus)
        cutoff = np.argmin(cutoff) if np.any(cutoff) else -1
        self.cutoff = cutoff * dt
        self.tau = self.acfint[cutoff]

        # Compute the diffusivity at the average position.
        self.position = np.mean(trajectories)
        self.diffusivity = np.var(trajectories, ddof=1) / self.tau


@numba.jit(nopython=True, fastmath=True)
def _ld_core(
    nsteps: int,
    mass: float,
    gamma: float,
    kt: float,
    x0: float,
    v0: float,
    dt: float,
    fext: numba.types.Callable,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate one-dimensional Langevin dynamics.

    Numba-optimized core function implementing the BAOAB integrator.

    :param nsteps: The number of simulation steps.
    :param mass: The mass of the particle
    :param gamma: The friction constant.
    :param kt: The thermal energy (Boltzmann constant times temperature).
    :param x0: The initial position.
    :param v0: The initial velocity.
    :param dt: The time step.
    :param fext: The force exerted by the external potential.
    :param rng: A random number generator.

    :return: The simulated positions and velocities.
    """
    # BAOAB is formulated in terms of a damping rate.
    gamma = gamma / mass
    x = np.empty(nsteps + 1, dtype=np.float64)
    v = np.empty(nsteps + 1, dtype=np.float64)
    x[0] = x0
    v[0] = v0
    c1 = np.exp(-gamma * dt)
    c3 = np.sqrt(kt * (1 - c1**2))
    c3m = c3 / np.sqrt(mass)
    dtm = dt / mass
    r = rng.normal(size=nsteps)
    for i in range(1, nsteps + 1):
        v[i] = v[i - 1] + 0.5 * fext(x[i - 1]) * dtm
        x[i] = x[i - 1] + 0.5 * v[i] * dt
        v[i] = c1 * v[i] + c3m * r[i - 1]
        x[i] = x[i] + 0.5 * v[i] * dt
        v[i] = v[i] + 0.5 * fext(x[i]) * dtm
    return x, v


@numba.jit(nopython=True, fastmath=True)
def _ld_od_core(
    nsteps: int,
    gamma: float,
    kt: float,
    x0: float,
    dt: float,
    fext: numba.types.Callable,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate one-dimensional Langevin dynamics in the limit of strong damping.

    Numba-optimized core function implementing the Euler-Maruyama method.

    :param nsteps: The number of simulation steps.
    :param gamma: The damping constant.
    :param kt: The thermal energy (Boltzmann constant times temperature).
    :param x0: The initial position.
    :param dt: The time step.
    :param fext: The force exerted by the external potential.
    :param rng: A random number generator.

    :return: The simulated positions and velocities.
    """
    x = np.empty(nsteps + 1, dtype=np.float64)
    v = np.empty(nsteps + 1, dtype=np.float64)
    x[0] = x0
    r = rng.normal(loc=0.0, scale=np.sqrt(2.0 * kt * dt / gamma), size=nsteps)
    dtg = dt / gamma
    for i in range(1, nsteps + 1):
        x[i] = x[i - 1] + fext(x[i - 1]) * dtg + r[i - 1]
    return x, v


@numba.jit(nopython=True)
def _no_force(_: float) -> float:
    """Return no force."""
    return 0.0


def _ld_validate(nsteps: int, mass: float, diff: float, kt: float, dt: float):
    """
    Validate the parameters for simulating Langevin dynamics.

    Ensures that all input parameters meet the required bounds and stability conditions.

    :param nsteps: The number of simulation steps.
    :param mass: The mass of the particle, must be non-negative.
    :param diff: The diffusivity of the particle, must be non-negative.
    :param kt: The thermal energy (Boltzmann constant times temperature), must be non-negative.
    :param dt: The time step, must be non-negative and satisfy stability constraints.

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


def ld(
    nsteps: int,
    *,
    mass: float = 1.0,
    diff: float = 1.0,
    kt: float = 1.0,
    x0: float = 0.0,
    v0: float = 0.0,
    dt: float = 1.0,
    od: bool = False,
    fext: Optional[numba.types.Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
    """
    Simulate one-dimensional Langevin dynamics.

    :param nsteps: The number of simulation steps.
    :param mass: The mass.
    :param diff: The diffusion constant.
    :param kt: The thermal energy (Boltzmann constant times temperature).
    :param x0: The initial position.
    :param v0: The initial velocity.
    :param dt: The time step.
    :param od: Simulate in the high-friction limit?
    :param fext: A function returning an external force. Defaults to `None`.
    :param rng: A random number generator. Defaults to `np.random.default_rng()`.

    :return: The simulated trajectory (positions, velocities).
    """

    # Validate the simulation parameters.
    _ld_validate(nsteps=nsteps, mass=mass, diff=diff, kt=kt, dt=dt)

    # Initialize the optional parameters.
    rng = rng or np.random.default_rng()
    fext = fext or _no_force

    # Simulate Langevin dynamics in an external potential.
    gamma = kt / diff
    if od:
        x = _ld_od_core(
            nsteps=nsteps,
            gamma=gamma,
            kt=kt,
            x0=x0,
            dt=dt,
            fext=fext,
            rng=rng,
        )
        return x
    else:
        x, v = _ld_core(
            nsteps=nsteps,
            mass=mass,
            gamma=gamma,
            kt=kt,
            x0=x0,
            v0=v0,
            dt=dt,
            fext=fext,
            rng=rng,
        )
        return x, v


def grw(
    nsteps: int,
    *,
    diff: float = 1.0,
    x0: float = 0.0,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Simulate a one-dimensional Gaussian random walk approximating Brownian motion.

    :param nsteps: The number of simulation steps.
    :param diff: The diffusion constant.
    :param x0: The initial position.
    :param dt: The time step.
    :param rng: The random number generator. Defaults to `np.random.default_rng()`.

    :return: The simulated trajectory (positions).
    """
    # Initialize a RNG.
    rng = rng or np.random.default_rng()
    # Simulate a Gaussian random walk.
    g = rng.normal(loc=0.0, scale=np.sqrt(2.0 * diff * dt), size=nsteps + 1)
    g[0] = x0
    x = np.cumsum(g)
    return x


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
    Compute the one-dimensional, mean squared displacement (MSD) of the input trajectory.

    For an efficient computation, the mean squared displacement
        MSD(n) = <[x(n) - x(0)]^2>
    is separated into
        MSD(n) = MSSQ(n) - 2.0 * TCF(n),
    where
        TCF(n) = <x(n) * x(0)>
    is the time correlation function (TCF) and
        MSSQ(n) = <x(n)^2 + x(0)^2>
    is the time-lagged, mean sum of squares (MSSQ) of the input trajectory.

    The TCF is computed efficiently using the Fast Fourier Transform (FFT) and the mean sum of squares is computed
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
