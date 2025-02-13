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
            self.t = t
            self.ref = ref
            self._samples = []

        def add(self, sample):
            self._samples.append(sample)

        @property
        def samples(self):
            return np.transpose(self._samples)

        @property
        def mean(self):
            return np.mean(self._samples, axis=0)


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
                self.x.add(x)
                self.msd.add(msd)

        def plot(self):
            nrows, ncols = 1, 2
            fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

            sample_style = {"color": "C1", "alpha": 0.5}
            mean_style = {"color": "C2", "label": "sample mean"}
            ref_style = {"color": "C0", "ls": "--", "label": "reference"}

            for ax in axes:
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.yaxis.set_major_locator(plt.MaxNLocator(6))

            ax = axes[0]
            ax.plot(self.x.t, self.x.samples, **sample_style)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$x$")

            ax = axes[1]
            ax.plot(self.msd.t, self.msd.samples, **sample_style)
            ax.plot(self.msd.t, self.msd.mean, **mean_style)
            ax.plot(self.msd.t, self.msd.ref, **ref_style)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\left<\left[x(t)-x(0)\right]^2\right>$")
            ax.legend()

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


@app.cell
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
                self.x.add(x)
                self.v.add(v)
                self.msd.add(msd)
                self.cv.add(cv)

        def plot(self):
            nrows, ncols = 2, 2
            fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

            sample_style = {"color": "C1", "alpha": 0.5}
            mean_style = {"color": "C2", "label": "sample mean"}
            ref_style = {"color": "C0", "ls": "--", "label": "reference"}

            for ax in axes.flat:
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.yaxis.set_major_locator(plt.MaxNLocator(6))

            ax = axes[0, 0]
            ax.plot(self.x.t, self.x.samples, **sample_style)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$x$")

            ax = axes[1, 0]
            ax.plot(self.v.t, self.v.samples, **sample_style)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$v$")

            ax = axes[0, 1]
            ax.plot(self.msd.t, self.msd.samples, **sample_style)
            ax.plot(self.msd.t, self.msd.mean, **mean_style)
            ax.plot(self.msd.t, self.msd.ref, **ref_style)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\left<\left[x(t)-x(0)\right]^2\right>$")
            ax.legend()

            ax = axes[1, 1]
            ax.plot(self.cv.t, self.cv.samples, **sample_style)
            ax.plot(self.cv.t, self.cv.mean, **mean_style)
            ax.plot(self.cv.t, self.cv.ref, **ref_style)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\left<v(t)v(0)\right>$")
            ax.legend()
            ax.set_yscale("log")

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


app._unparsable_cell(
    r"""
    def ld_ho_samples(
        nsamples, nsteps, nmsds, nptcfs, nvtcfs, dt, mass, kt, diff, fcon, seed=42
    ):
        rng = np.random.default_rng(seed)

        t = np.arange(nsteps + 1) * dt
        tmsd = np.arange(nmsds) * dt
        tptcf = np.arange(nptcfs) * dt
        tvtcf = np.arange(nvtcfs) * dt
        s = np.linspace(0.0, 500.0, 1000)

        xsamples = []
        vsamples = []
        msds = []
        ptcfs = []
        vtcfs = []
        laplaces = []
        for _ in range(nsamples):
            x, v = mvdlib.diffusion.ld(
                nsteps,
                mass=mass,
                diff=diff,
                kt=kt,
                dt=dt,
                rng=rng,
                fext=numba.njit(lambda x: -fcon * x),
            )
            msd = mvdlib.diffusion.msd(x, maxsteps=nmsds)
            ptcf = mvdlib.stats.tcf(x, nc=nptcfs)
            vtcf = mvdlib.stats.tcf(v, nc=nvtcfs)

            laplace = mvdlib.diffusion.laplace(vtcf, , s)
            xsamples.append(x)
            vsamples.append(v)
            msds.append(msd)
            ptcfs.append(ptcf)
            vtcfs.append(vtcf)
            laplaces.append(laplace)

        damp = kt / diff
        tau = mass / damp

        mean_msd = np.mean(msds, axis=0)
        limit_msd = 2.0 * kt / fcon

        mean_ptcf = np.mean(ptcfs, axis=0)
        omega_0 = np.sqrt(fcon / mass)
        omega_r_sq = omega_0**2 - (kt / (2.0 * mass * diff)) ** 2
        if omega_r_sq == 0.0:
            expected_ptcf = (
                (kt / fcon) * (1.0 + omega_0 * tptcf) * np.exp(-omega_0 * tptcf)
            )
            expected_vtcf = (
                (kt / mass)
                * (1.0 - 0.5 * damp / mass * tvtcf)
                * np.exp(-0.5 * damp / mass * tvtcf)
            )
        elif omega_r_sq < 0.0:
            omega_r = np.sqrt(-omega_r_sq)
            expected_ptcf = (kt / fcon) * np.exp(-fcon / damp * tptcf)
            expected_vtcf = (
                (kt / mass)
                * np.exp(-0.5 * damp / mass * tvtcf)
                * (
                    np.cosh(omega_r * tvtcf)
                    - damp / (2.0 * mass * omega_r) * np.sinh(omega_r * tvtcf)
                )
            )
        else:
            omega_r = np.sqrt(omega_r_sq)
            expected_ptcf = (
                (kt / fcon)
                * np.exp(-0.5 * tptcf / tau)
                * (
                    np.cos(omega_r * tptcf)
                    + np.sin(omega_r * tptcf) / (2.0 * omega_r * tau)
                )
            )
            expected_vtcf = (
                (kt / mass)
                * np.exp(-0.5 * damp / mass * tvtcf)
                * (
                    np.cos(omega_r * tvtcf)
                    - damp / (2.0 * mass * omega_r) * np.sin(omega_r * tvtcf)
                )
            )
        mean_vtcf = np.mean(vtcfs, axis=0)
        expected_msd = limit_msd - 2.0 * expected_ptcf

        mean_laplace = np.mean(laplaces, axis=0)
        expected_laplace = (kt / mass) * s / (s**2 + damp / mass * s + fcon / mass)

        return SimpleNamespace(
            {
                \"t\": t,
                \"xsamples\": xsamples,
                \"vsamples\": vsamples,
                \"tmsd\": tmsd,
                \"msds\": msds,
                \"mean_msd\": mean_msd,
                \"limit_msd\": limit_msd,
                \"expected_msd\": expected_msd,
                \"tptcf\": tptcf,
                \"tvtcf\": tvtcf,
                \"ptcfs\": ptcfs,
                \"vtcfs\": vtcfs,
                \"mean_ptcf\": mean_ptcf,
                \"mean_vtcf\": mean_vtcf,
                \"expected_ptcf\": expected_ptcf,
                \"expected_vtcf\": expected_vtcf,
                \"laplaces\": laplaces,
                \"mean_laplace\": mean_laplace,
                \"expected_laplace\": expected_laplace,
                \"s\": s,
            }
        )


    def _get_datasets():
        mass = 18.0
        kt = 2.5
        diff = 3.0e-3
        kcrit = mass * (kt / (2.0 * mass * diff)) ** 2

        datasets = []

        for fac, nc, dt in [
            (0.1, 5000, 0.002),
            (1.0, 1000, 0.001),
            (10.0, 3000, 0.0002),
        ]:
            nsteps = nc * 100
            fcon = fac * kcrit
            data = ld_ho_samples(
                nsamples=10,
                nsteps=nsteps,
                nmsds=nc,
                nptcfs=nc,
                nvtcfs=int(0.5 / dt),
                dt=dt,
                mass=mass,
                kt=kt,
                diff=diff,
                fcon=fac * kcrit,
            )
            datasets.append((fac, fcon, data))

        return datasets


    def ld_ho_plot(datasets):
        nrows, ncols = 4, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        for i, (fac, fcon, data) in enumerate(datasets):
            for msd in data.msds:
                axes[0, i].plot(data.tmsd, msd, color=\"C1\", alpha=0.5)
            for ptcf in data.ptcfs:
                axes[1, i].plot(data.tptcf, ptcf, color=\"C1\", alpha=0.5)
            for vtcf in data.vtcfs:
                axes[2, i].plot(data.tvtcf, vtcf, color=\"C1\", alpha=0.5)
            for laplace in data.laplaces:
                axes[3, i].plot(data.s, laplace, color=\"C1\", alpha=0.5)

            if fac < 1.0:
                title = \"strong damping\"
            elif fac > 1.0:
                title = \"weak damping\"
            else:
                title = \"critical damping\"
            axes[0, i].set_title(title)

            axes[0, i].plot(
                data.tmsd, data.mean_msd, color=\"C2\", label=\"sample mean\"
            )
            axes[0, i].plot(
                data.tmsd,
                data.expected_msd,
                color=\"C0\",
                ls=\"--\",
                label=\"expected\",
            )

            axes[1, i].plot(
                data.tptcf, data.mean_ptcf, color=\"C2\", label=\"sample mean\"
            )
            axes[1, i].plot(
                data.tptcf,
                data.expected_ptcf,
                color=\"C0\",
                ls=\"--\",
                label=\"expected\",
            )

            axes[2, i].plot(
                data.tvtcf, data.mean_vtcf, color=\"C2\", label=\"sample mean\"
            )
            axes[2, i].plot(
                data.tvtcf,
                data.expected_vtcf,
                color=\"C0\",
                ls=\"--\",
                label=\"expected\",
            )

            axes[3, i].plot(
                data.s, data.mean_laplace, color=\"C2\", label=\"sample mean\"
            )
            axes[3, i].plot(
                data.s,
                data.expected_laplace,
                color=\"C0\",
                ls=\"--\",
                label=\"expected\",
            )
            min, max = np.min(data.expected_laplace), np.max(data.expected_laplace)
            axes[3, i].set_ylim(min - 0.02 * (max - min), max + 0.02 * (max - min))

        for ax in axes.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax.legend()

        for ax in axes[:2, :].flat:
            ax.set_xlabel(r\"$t$\")

        for ax in axes[3, :].flat:
            ax.set_xlabel(r\"$s$\")

        axes[0, 0].set_ylabel(r\"$\left<\left[x(t)-x(0)\right]^2\right>$\")
        axes[1, 0].set_ylabel(r\"$\left<x(t)x(0)\right>$\")
        axes[2, 0].set_ylabel(r\"$\left<v(t)v(0)\right>$\")
        axes[3, 0].set_ylabel(
            r\"$\mathcal{{L}}\left\lbrace\left<v(t)v(0)\right>\right\rbrace(s)$\"
        )

        return fig


    mo.md(
        rf\"\"\"
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

        {mo.as_html(ld_ho_plot(_get_datasets()))}    
        \"\"\"
    )
    """,
    name="_",
    column=None, disabled=False, hide_code=True
)


@app.cell(hide_code=True)
def _():
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numba

    import mvdlib.diffusion
    import mvdlib.plots
    import mvdlib.stats
    import mvdlib.style

    plt.style.use("mvdlib.style.default")
    FIGSIZE = mvdlib.plots.FigSize()
    return FIGSIZE, SimpleNamespace, mo, mvdlib, np, numba, plt


if __name__ == "__main__":
    app.run()
