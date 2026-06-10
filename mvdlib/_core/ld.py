import numba
import numpy as np
from numpy.typing import NDArray

from mvdlib.types import NumbaScalarFloatFunc


@numba.njit(fastmath=True)
def ld(
    force: NumbaScalarFloatFunc,
    friction: NumbaScalarFloatFunc,
    nsteps: int,
    dt: float,
    mass: float,
    kt: float,
    x0: float,
    v0: float,
    save_freq: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate one-dimensional underdamped Langevin dynamics.

    Implements a BAOAB discretization of

        m dv = force(x) dt - friction(x) v dt + sqrt(2 kT friction(x)) dW,
        dx = v dt.

    The returned trajectory includes the initial state.
    No argument validation is performed.
    """
    ns = nsteps // save_freq + 1
    xs = np.empty(ns, dtype=np.float64)
    vs = np.empty(ns, dtype=np.float64)

    half_dtm = 0.5 * dt / mass
    half_dt = 0.5 * dt
    sigma_v = np.sqrt(kt / mass)
    noise = rng.standard_normal(size=nsteps)

    x = x0
    v = v0
    xs[0] = x
    vs[0] = v
    save_idx = 1

    for i in range(nsteps):
        v += force(x) * half_dtm
        x += v * half_dt

        gamma = friction(x) / mass
        a = np.exp(-gamma * dt)
        v = a * v + sigma_v * np.sqrt(1.0 - a * a) * noise[i]

        x += v * half_dt
        v += force(x) * half_dtm
        if (i + 1) % save_freq == 0:
            xs[save_idx] = x
            vs[save_idx] = v
            save_idx += 1

    return xs, vs
