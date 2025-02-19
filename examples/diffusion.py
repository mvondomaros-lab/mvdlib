import marimo

__generated_with = "0.11.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Diffusion""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Homogeneous Media""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Gaussian random walk""")
    return


@app.cell(hide_code=True)
def _(FIGSIZE, mo, mvdlib, np, plt):
    class Samples:
        def __init__(self, t, ref=None):
            self._t = t
            self._ref = ref
            self._samples = []

        def append(self, sample):
            self._samples.append(sample)

        def plot(
            self,
            ax,
            xlabel=None,
            ylabel=None,
            mean=True,
            yscale=None,
            linthresh=1.0e-2,
        ):
            sample_style = {"color": "C1", "alpha": 0.5}
            mean_style = {"color": "C2", "label": "sample mean"}
            ref_style = {"color": "C0", "ls": "--", "label": "reference"}

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

            ax.plot(self._t, np.transpose(self._samples), **sample_style)

            if mean:
                ax.plot(self._t, np.mean(self._samples, axis=0), **mean_style)
                ax.legend()

            if self._ref is not None:
                ax.plot(self._t, self._ref, **ref_style)
                ax.legend()

            if yscale is None:
                pass
            elif yscale == "symlog":
                ax.set_yscale("symlog", linthresh=linthresh)
            else:
                ax.set_yscale(yscale)


    class GRWData:
        def __init__(
            self,
            nsamples=10,
            nsteps=10000,
            nmsd=100,
            dt=0.001,
            diff=3.0e-3,
            seed=42,
        ):
            rng = np.random.default_rng(seed)

            t = np.arange(nsteps + 1) * dt
            self.x = Samples(t)

            t = np.arange(nmsd) * dt
            ref = 2.0 * diff * t
            self.msd = Samples(t, ref)

            for _ in range(nsamples):
                x = mvdlib.diffusion.grw(nsteps, diff=diff, dt=dt, rng=rng)
                msd = mvdlib.diffusion.msd(x, maxsteps=nmsd)
                self.x.append(x)
                self.msd.append(msd)

        def plot(self):
            nrows, ncols = 1, 2
            fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

            self.x.plot(axes[0], xlabel="$t$", ylabel="$x$", mean=False)
            self.msd.plot(
                axes[1],
                xlabel="t",
                ylabel=r"$\left<\left[x(t)-x(0)\right]^2\right>$",
            )

            return mo.as_html(fig)


    mo.md(
        rf"""
        If Brownian motion $B(t)$ is sampled at discrete time steps, $t_n = n\Delta{{}}t$, it becomes indistinguishable from a properly scaled Gaussian random walk,

        $$
        \begin{{align*}}
        x_n &= \sum_{{i=0}}^n g_i \,, \\
        g_i &\sim \mathcal{{N}}(0, \sqrt{{2D\Delta{{}}t}}) \,.
        \end{{align*}}
        $$

        This is because the increments of Brownian motion over discrete intervals $\Delta{{}}t$,
        are i.i.d. Gaussian random variables, just like the steps in a Gaussian random walk.

        $$
        B(t_{{n+1}}) - B(t_{{n}}) \sim \mathcal{{N}}(0, \sqrt{{2D\Delta{{}}t}})\,.
        $$

        We can use this fact to simulate Brownianian motion, by coarse-graining in time.

        The mean squared displacement (MSD) is a linear function with time.

        $$
        \left<\left[x(t)-x(0)\right]^2\right> = 2Dt
        $$

        Furthermore, a Gaussian random walk approaches Brownian motion, in the limit of small time steps,
        while maintaining proper scaling.
        This is [Donsker's theorem](https://en.wikipedia.org/wiki/Donsker%27s_theorem).

        {GRWData().plot()}
        """
    )
    return GRWData, Samples


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Langevin Dynamics""")
    return


@app.cell(hide_code=True)
def _(FIGSIZE, Samples, mo, mvdlib, np, plt):
    class LDData:
        def __init__(
            self,
            nsamples=10,
            nsteps=10000,
            nmsd=100,
            ncv=100,
            dt=0.001,
            mass=18.0,
            kt=2.5,
            diff=3.0e-3,
            seed=42,
        ):
            rng = np.random.default_rng(seed)

            gamma = kt / diff
            tau = mass / gamma

            t = np.arange(nsteps + 1) * dt
            self.x = Samples(t)
            self.v = Samples(t)

            t = np.arange(nmsd) * dt
            ref = 2.0 * diff * (t - tau * (1.0 - np.exp(-t / tau)))
            self.msd = Samples(t, ref)

            t = np.arange(ncv) * dt
            ref = (kt / mass) * np.exp(-t / tau)
            self.cv = Samples(t, ref)

            for _ in range(nsamples):
                x, v = mvdlib.diffusion.ld(
                    nsteps, mass=mass, diff=diff, kt=kt, dt=dt, rng=rng
                )
                msd = mvdlib.diffusion.msd(x, maxsteps=nmsd)
                cv = mvdlib.stats.tcf(v, nc=ncv)
                self.x.append(x)
                self.v.append(v)
                self.msd.append(msd)
                self.cv.append(cv)

        def plot(self):
            nrows, ncols = 2, 2
            fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

            self.x.plot(axes[0, 0], xlabel="$t$", ylabel="$x$", mean=False)
            self.v.plot(axes[1, 0], xlabel="$t$", ylabel="$v$", mean=False)
            self.msd.plot(
                axes[0, 1],
                xlabel="$t$",
                ylabel=r"$\left<\left[x(t)-x(0)\right]^2\right>$",
            )
            self.cv.plot(
                axes[1, 1],
                xlabel="$t$",
                ylabel=r"$\left<v(t)v(0)\right>$",
                yscale="log",
            )

            return mo.as_html(fig)


    mo.md(
        rf"""
        The Langevin equation is a stochastic differential equation that describes how a system evolves when subjected to both deterministic and fluctuating (random) forces.

        $$
        \begin{{align}}
            \frac{{dx}}{{dt}} &= v \\
            \frac{{dv}}{{dt}} &= -\frac{{\lambda}}{{m}} v + \frac{{\eta(t)}}{{m}}
        \end{{align}}
        $$

        Here, $\lambda$ is a damping constant and $\eta(t)$ is random force defined in terms of its first and second moments.

        $$
        \begin{{align}}
            \left<\eta(t)\right> &= 0\\
            \left<\eta(t)\eta(t^\prime)\right> &= 2\lambda k_\mathsf{{B}}T\delta(t-t^\prime)
        \end{{align}}
        $$

        The Langevin equation can be easily solved for the average velocity.

        $$
        \left<v(t)\right> = v_0\exp{{\left(-\frac{{t}}{{\tau}}\right)}}, \quad \tau = \frac{{m}}{{\lambda}}
        $$

        It can also be written as an Ornstein--Uhlenbeck (OU) process.

        $$
        \begin{{align}}
            dv &= -\theta v dt + \sigma dB(t) \\
            \theta &= \frac{{1}}{{\tau}} = \frac{{\lambda}}{{m}} \\
            \sigma &= \frac{{\sqrt{{2\lambda k_\mathsf{{B}}T}}}}{{m}}
        \end{{align}}
        $$

        Here, $\theta$ is the mean reversion rate, $\sigma$ is the diffusion constant (of the OU process), and $dB(t)$ is the presumed derivatve of Brownian motion.

        The dynamics of an OU process can be solved analytically.
        In particular, we can derive expressions for the velocity time correlation function and the mean squared displacement, given that the system is initially
        in equilibrium.

        $$
        \begin{{align}}
            \left<v(t)v(0)\right> &= \frac{{k_\mathsf{{B}}T}}{{m}} \exp{{\left(-\frac{{t}}{{\tau}}\right)}} \\
            \left<\left[x(t)-x(0)\right]^2\right> &= \frac{{2k_\mathsf{{B}}T}}{{\lambda}} \left[t - \tau\left(1 - \exp{{\left({{-\frac{{t}}{{\tau}}}}\right)}}\right)\right]
        \end{{align}}
        $$

        In the limit of long lag times, we recover the Einstein relation.

        $$
        \lim_{{t\to\infty}}\left<\left[x(t)-x(0)\right]^2\right> = 2Dt, \quad D = \frac{{k_\mathsf{{B}}T}}{{\lambda}}
        $$


        Langevin dynamics can be simulated using finite differences.

        $$
        \begin{{align}}
            v_{{n+1}} &= v_n - \theta v_{{n}} \Delta{{t}} + \sigma g_n\sqrt{{\Delta t}} \\
            x_{{n+1}} &= x_n + v_n \Delta{{t}}
        \end{{align}}
        $$

        {LDData().plot()}
        """
    )
    return (LDData,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Harmonic Potential""")
    return


@app.cell(hide_code=True)
def _(FIGSIZE, Samples, mo, mvdlib, np, numba, plt):
    class LDHOData:
        def __init__(
            self,
            nsamples=10,
            ncx=1000,
            ncv=500,
            ns=500,
            dt=0.001,
            mass=18.0,
            kt=2.5,
            diff=3.0e-3,
            cfcscale=1.0,
            seed=42,
        ):
            rng = np.random.default_rng(seed)

            gamma = kt / diff
            tau = mass / gamma
            cfc = mass * (gamma / (2.0 * mass)) ** 2
            fc = cfc * cfcscale

            omg0_sq = fc / mass
            omg0 = np.sqrt(omg0_sq)
            omgr_sq = omg0**2 - (kt / (2.0 * mass * diff)) ** 2
            omgr = np.sqrt(np.abs(omgr_sq))

            nsteps = 100 * ncx

            # x, v
            t = np.arange(nsteps + 1) * dt
            self.x = Samples(t)
            self.v = Samples(t)

            # PACF
            t = np.arange(ncx) * dt
            ref = kt / fc
            if omgr_sq == 0.0:
                ref *= (1.0 + omg0 * t) * np.exp(-omg0 * t)
            elif omgr_sq < 0.0:
                ref *= np.exp(-fc / gamma * t)
            else:
                ref *= np.exp(-0.5 * t / tau) * (
                    np.cos(omgr * t) + np.sin(omgr * t) / (2.0 * omgr * tau)
                )
            ref_cx = ref
            self.cx = Samples(t, ref_cx)

            # MSD
            t = np.arange(ncx) * dt
            ref = 2.0 * kt / fc - 2.0 * ref_cx
            self.msd = Samples(t, ref)

            # VACF
            t = np.arange(ncv) * dt
            tcv = t
            ref = (kt / mass) * np.exp(-0.5 * gamma / mass * t)
            if omgr_sq == 0.0:
                ref *= 1.0 - 0.5 * gamma / mass * t
            elif omgr_sq < 0.0:
                ref *= np.cosh(omgr * t) - gamma / (2.0 * mass * omgr) * np.sinh(
                    omgr * t
                )
            else:
                ref *= np.cos(omgr * t) - gamma / (2.0 * mass * omgr) * np.sin(
                    omgr * t
                )
            self.cv = Samples(tcv, ref)

            # Laplace transform of VACF
            s = np.linspace(0.0, 200.0, ns)
            ref = (kt / mass) * s / (s**2 + gamma / mass * s + fc / mass)
            self.lcv = Samples(s, ref)

            for _ in range(nsamples):
                x, v = mvdlib.diffusion.ld(
                    nsteps,
                    mass=mass,
                    diff=diff,
                    kt=kt,
                    dt=dt,
                    fext=numba.njit(lambda x: -fc * x),
                    rng=rng,
                )
                msd = mvdlib.diffusion.msd(x, maxsteps=ncx)
                cx = mvdlib.stats.tcf(x, nc=ncx)
                cv = mvdlib.stats.tcf(v, nc=ncv)
                lcv = mvdlib.helpers.laplace(cv, tcv, s)
                self.x.append(x)
                self.v.append(v)
                self.msd.append(msd)
                self.cx.append(cx)
                self.cv.append(cv)
                self.lcv.append(lcv)

        def plot(self):
            nrows, ncols = 1, 4
            fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

            self.msd.plot(
                axes[0],
                xlabel="$t$",
                ylabel=r"$\left<\left[x(t)-x(0)\right]^2\right>$",
            )
            self.cx.plot(axes[1], xlabel="$t$", ylabel=r"$\left<x(t)x(0)\right>$")
            self.cv.plot(
                axes[2],
                xlabel="$t$",
                ylabel=r"$\left<v(t)v(0)\right>$",
                yscale="symlog",
            )
            self.lcv.plot(
                axes[3],
                xlabel="$s$",
                ylabel=r"$\mathcal{{L}}\left\lbrace\left<v(t)v(0)\right>\right\rbrace(s)$",
            )

            return mo.as_html(fig)


    mo.md(
        rf"""
        The Langevin equation can be extended to describe diffusion in a harmonic potential.

        $$
        \begin{{align}}
            \frac{{dx}}{{dt}} &= v \\
            \frac{{dv}}{{dt}} &=  -\omega_0^2x -\frac{{\lambda}}{{m}} v + \frac{{\eta(t)}}{{m}}, \quad \omega_0 = \sqrt{{\frac{{k}}{{m}}}}
        \end{{align}}
        $$

        If we define a resonance frequency $\omega_r$, we can distringuish between weak, critical, and strong damping.

        $$
            \omega_r^2 = \left|\omega_0^2 - \left(\frac{{\lambda}}{{2m}}\right)^2\right| = \left|\frac{{k}}{{m}} - \left(\frac{{k_\mathsf{{B}}T}}{{2mD}}\right)^2\right| 
        $$

        The damping is weak if $k>m\left(\frac{{\lambda}}{{2m}}\right)^2$ and strong if $k<m\left(\frac{{\lambda}}{{2m}}\right)^2$. The case of equality corresponds to critical damping.

        The position time correlation function can be evaluated analytically.

        $$
        \left<x(t)x(0)\right> = \frac{{k_\mathsf{{B}}T}}{{k}}\exp{{\left(-\frac{{\lambda}}{{2m}}t\right)}}\begin{{cases}}
        \times\left(\cos{{\omega_r t}} + \frac{{\lambda}}{{2m\omega_r}}\sin{{\omega_r t}}\right) && \text{{weak damping}}\\
        \times\left(1+\frac{{\lambda}}{{2m}}t\right) && \text{{critical damping}} \\
        \times\left(\cosh{{\omega_r t}} + \frac{{\lambda}}{{2m\omega_r}}\sinh{{\omega_r t}}\right) && \text{{strong damping}}\\
        \end{{cases}}
        $$

        Note, that equipartion demands $k\left<x^2\right>=m\left<v^2\right>=k_\mathsf{{B}}T$.

        It is then straightforward to evaluate the mean squared displacement and the velocity time correlation function.

        $$
        \begin{{align}}
        \left<\left[x(t)-x(0)\right]^2\right> &= 2\left<x^2\right> - \left<x(t)x(0)\right> \\
        \left<v(t)v(0)\right> &= \frac{{d^2}}{{dt^2}}\left<x(t)x(0)\right>\\
        \end{{align}}
        $$

        While Langevin dynamics in a potential is not an OU process anymore, it can be integrated similarly.

        $$
        \begin{{align}}
            v_{{n+1}} &= v_n + (-\omega_0^2x_n -\theta v_{{n}}) \Delta{{t}} + \sigma g_n \sqrt{{\Delta t}} \\
            x_{{n+1}} &= x_n + v_n \Delta{{t}}
        \end{{align}}
        $$

        ### Strong Damping

        {LDHOData(ncx=5000, ncv=500, cfcscale=0.1, dt=0.002).plot()}

        ### Critical Damping
        {LDHOData(cfcscale=1.0).plot()}

        ### Weak Damping
        {LDHOData(ncx=4000, ncv=4000, dt=0.0001, cfcscale=10.0).plot()}
        """
    )
    return (LDHOData,)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numba
    import numpy as np

    import mvdlib.diffusion
    import mvdlib.plots
    import mvdlib.stats
    import mvdlib.style

    plt.style.use("mvdlib.style.default")
    FIGSIZE = mvdlib.plots.FigSize()
    return FIGSIZE, mo, mvdlib, np, numba, plt


if __name__ == "__main__":
    app.run()
