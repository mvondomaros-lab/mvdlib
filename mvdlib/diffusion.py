from __future__ import annotations

import typing

import numba
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Collection, Optional, Sequence, Tuple

    from numpy.typing import NDArray


class USVTCFAnalysis:
    def __init__(
        self,
        trajectories: Collection[Sequence[NDArray[np.float64], NDArray[np.float64]]],
        nc: int,
        dt: float = 0.001,
    ):
        """
        Analyze the velocity time correlation functions of harmonically restrained MD simulations.

        :param trajectories: Input trajectories (positions, velocities).
        :param nc: The correlation length.
        :param dt: The simulation time step.

        :raises ValueError: For erroneous trajectory inputs.
        """
        import mvdlib.helpers
        import mvdlib.stats

        # Verify the trajectory input.
        # TODO: Handle arrays of different lengths.
        try:
            for x, v in trajectories:
                if (
                    not isinstance(x, np.ndarray)
                    or not isinstance(v, np.ndarray)
                    or x.ndim != 1
                    or x.shape != v.shape
                ):
                    raise ValueError(
                        "positions and velocities must be one-dimensional arrays of equal length"
                    )
        except TypeError:
            raise ValueError(
                "trajectories must be a collection of (position, velocity) sequences."
            )

        self.t = np.arange(nc) * dt
        ds = 1.0 / (nc * dt)
        smax = 1.0 / (2.0 * dt)
        self.s = np.arange(ds, smax, ds)
        self.x = np.mean([x for x, _ in trajectories])

        c_list = []
        ds_list = []
        ds_fit_list = []
        diff_list = []
        for x, v in trajectories:
            c = mvdlib.stats.tcf(v, nc=nc, shift=False, scale=False)
            lpl = mvdlib.helpers.laplace(c, self.t, self.s)
            varx = np.var(x, ddof=1)
            varv = np.var(v, ddof=1)
            ds = -(
                lpl
                * varx
                * varv
                / (lpl * (self.s * varx + varv / self.s) - varx * varv)
            )
            diff, ds_fit = USVTCFAnalysis._get_ds_fit(self.s, ds)

            c_list.append(c)
            ds_list.append(ds)
            ds_fit_list.append(ds_fit)
            diff_list.append(diff)

        self.c = np.mean(c_list, axis=0)
        self.ds = np.mean(ds_list, axis=0)
        self.ds_fit = np.mean(ds_fit_list, axis=0)
        self.diff = np.mean(diff_list)
        self.sem = np.std(diff_list, ddof=1) / np.sqrt(len(diff_list))

    @staticmethod
    def _get_ds_fit(s, ds):
        # Restrict the fit to regions of small curvature.
        ds_grad = np.gradient(ds, s)
        ds_grad2 = np.gradient(ds_grad, s)
        mask = np.abs((ds_grad2 - np.median(ds_grad2)) / ds_grad2) < 0.05
        s_mask = s[mask]
        ds_mask = ds[mask]
        c = np.polyfit(s_mask, ds_mask, 4)
        return c[-1], np.poly1d(c)(s)


class USPTCFAnalysis:
    def __init__(
        self,
        trajectories: Collection[NDArray[np.float64]],
        nc: int,
        n0: int,
        dt: float = 1.0,
    ):
        """
        Analyze the position time correlation functions of harmonically restrained MD simulations.

        :param trajectories: Input trajectories (positions).
        :param nc: The correlation length.
        :param n0: The number of time steps, after which the PTCFs have converged to zero.
        :param dt: The simulation time step.

        :raises ValueError: For erroneous trajectory inputs.
        """
        from scipy.integrate import cumulative_simpson

        import mvdlib.stats

        # Verify the trajectory input.
        # TODO: Handle arrays of different lengths.
        try:
            for x in trajectories:
                if not isinstance(x, np.ndarray) or x.ndim != 1:
                    raise ValueError("positions must be one-dimensional arrays")
        except TypeError:
            raise ValueError("trajectories must be a collection of position arrays.")

        self.t = np.arange(nc) * dt
        self.x = np.mean(trajectories)

        c_list = []
        c_int_list = []
        taux_list = []
        diff_list = []
        for x in trajectories:
            c = mvdlib.stats.tcf(x, nc=nc, shift=True, scale=True)
            c_int = cumulative_simpson(y=c, x=self.t, initial=0.0)
            taux = np.mean(c_int[n0:])
            varx = np.var(x, ddof=1)
            diff = varx / taux

            c_list.append(c)
            c_int_list.append(c_int)
            taux_list.append(taux)
            diff_list.append(diff)

        self.c = np.mean(c_list, axis=0)
        self.c_int = np.mean(c_int_list, axis=0)
        self.taux = np.mean(taux_list)
        self.diff = np.mean(diff_list)
        self.sem = np.std(diff_list, ddof=1) / np.sqrt(len(diff_list))


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
