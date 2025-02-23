import marimo

__generated_with = "0.11.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Workflow for Computing Position-Dependent Diffusion Coefficients from Umbrella Sampling""")
    return


@app.cell(hide_code=True)
def _(FIGSIZE, mo, mvdlib, np, numba, plt):
    class BrownianMotion:
        def __init__(
            self,
            nsteps=1000,
            mass=18.0,
            diff=3.0e-3,
            kt=2.5,
            x0=10.0,
            dt=0.001,
            cfcscale=1.0,
            seed=42,
        ):
            gamma = kt / diff
            cfc = gamma**2 / (4.0 * mass)
            fc = cfcscale * cfc
            rng = np.random.default_rng(seed)
            self.t = np.arange(nsteps + 1) * dt
            self.x, self.v = mvdlib.diffusion.ld(
                nsteps,
                x0=x0,
                mass=mass,
                diff=diff,
                kt=kt,
                dt=dt,
                fext=numba.njit(lambda x: -fc * (x - x0)),
                rng=rng,
            )


    nsteps = 100000
    diff = 3.0e-3
    dt = 0.001
    seeds = np.arange(50)
    samples = [
        BrownianMotion(nsteps=nsteps, diff=diff, dt=dt, seed=seed, cfcscale=10.0)
        for seed in seeds
    ]


    def _plot(samples):
        nrows, ncols = 2, 1
        fig, axes = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True
        )

        for sample in samples:
            axes[0].plot(sample.t, sample.x, color="C1", alpha=0.5)
        axes[0].set_ylabel(r"$x\ /\ \mathsf{nm}$")
        axes[0].yaxis.set_major_locator(plt.MaxNLocator(5))

        for sample in samples:
            axes[1].plot(sample.t, sample.v, color="C1", alpha=0.5)
        axes[1].set_xlabel(r"$t\ /\ \mathsf{ps}$")
        axes[1].set_ylabel(r"$v\ /\ \mathsf{nm\,ps^{-1}}$")
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(5))
        axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))

        return mo.as_html(fig)


    mo.md(
        rf"""
        ## Preparation

        - Run multiple MD simulations and save the positions and velocities of the diffusing particle(s).
        - If only one particle is of interest, run multiple independent simulations.
        - Here, we fake MD simulations by simulating Brownian motion in a harmonic potential.
            In these simulations, a particle diffuses with a known diffusivity $D={diff}\ \mathsf{{nm^2/ps}}$.
            Our job is to recover this value.
        {_plot(samples)}
        """
    )
    return BrownianMotion, diff, dt, nsteps, samples, seeds


@app.cell(hide_code=True)
def _(FIGSIZE, dt, mo, mvdlib, plt, samples):
    def _plot(ptcf_analysis):
        nrows, ncols = 2, 1
        fig, axes = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True
        )

        ax = axes[0]
        ax.axhline(0.0, color="C0", ls="--")
        ax.plot(ptcf_analysis.t, ptcf_analysis.c, color="C1")
        ax.set_ylabel(r"$C_x(t)$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        ax = axes[1]
        ax.axhline(ptcf_analysis.taux, color="C0", ls="--", label=r"$\tau_X$")
        ax.plot(ptcf_analysis.t, ptcf_analysis.c_int, color="C1")
        ax.legend()
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(r"$\int_0^tC_x(t^\prime)dt^\prime\ /\ \mathsf{ps}$")
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        return mo.as_html(fig)


    ncx = 500
    n0 = ncx // 2
    ptcf_analysis = mvdlib.diffusion.USPTCFAnalysis(
        trajectories=[sample.x for sample in samples], nc=ncx, n0=n0, dt=dt
    )

    mo.md(
        rf"""
        ## Position Time Correlation Function (PTCF) Analysis

        - The position time correlation function (PTCF) is defined as follows.

        $$
            C_x(t) = \frac{{\left<\delta x(t)\delta x(0)\right>}}{{\langle\left(\delta x\right)^2\rangle}}
        $$


        - Make sure that the PTCF decays to zero, or alternativey, that its integral reaches a plateau.
            The plateau value corresponds to the correlation time  $\tau_x$ that we wish to determine.

            $$
            \tau_x = \int_0^\infty C_x(t) dt
            $$

        - `USPTCFAnalysis` determines the correlation time $\tau_x$ by averaging over the cumulative integral of the position time correlation function for times $t>t_0$, where $t_0$ is the time after which the PTCF is zero. Make sure that this is indeed the plateau value of the integral. Here, we obtain $\tau_x = {ptcf_analysis.taux:.3g}\ \mathsf{{ps}}$.

        - The position-dependent diffusion coefficient is computed as follows.
            Here, we obtain $D = ({ptcf_analysis.diff:.3g} \pm {ptcf_analysis.sem * 1.96:.3g})\ \mathsf{{nm^2\,ps^{{-1}}}}$ at $x = {ptcf_analysis.x:.3g}\ \mathsf{{nm}}$.

            $$
            D(\left<x\right>) = \frac{{\tau_x}}{{\langle\left(\delta x\right)^2\rangle}}
            $$ 

        {_plot(ptcf_analysis)}
        """
    )
    return n0, ncx, ptcf_analysis


@app.cell(hide_code=True)
def _(FIGSIZE, dt, mo, mvdlib, plt, samples):
    def _plot(vtcf_analysis):
        nrows, ncols = 1, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax = axes[0]
        ax.axhline(0.0, color="C0", ls="--")
        ax.plot(vtcf_analysis.t, vtcf_analysis.c, color="C1")
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(r"$C_v(t)\ /\ \mathsf{nm^2\,ps^{-2}}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        ax = axes[1]
        ax.plot(vtcf_analysis.s, vtcf_analysis.ds, color="C1")
        ax.plot(
            vtcf_analysis.s,
            vtcf_analysis.ds_fit,
            color="C0",
            ls="--",
            label="extrapolation",
        )
        ymax = vtcf_analysis.ds.max()
        ax.set_ylim(-0.02 * ymax, 1.02 * ymax)
        ax.legend()
        ax.set_xlabel(r"$s\ /\ \mathsf{ps^{-1}}$")
        ax.set_ylabel(r"$D(s)\ /\ \mathsf{nm^2\,ps^{-1}}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        return mo.as_html(fig)


    ncv = 500
    vtcf_analysis = mvdlib.diffusion.USVTCFAnalysis(
        trajectories=[(sample.x, sample.v) for sample in samples], nc=ncv, dt=dt
    )

    mo.md(
        rf"""
        ## Velocity Time Correlation Function (VTCF) Analysis

        - The velocity time correlation function (VTCF) is defined as follows.

        $$
            C_v(t) = \left<\delta v(t)\delta v(0)\right>
        $$


        - Make sure that the PTCF decays to zero.
        - `USVTCFAnalysis` computes the Laplace transform of the VTCF, $L_v(s)$, and a frequency-dependent diffusivity $D(s)$. Computations are limited to the range $s\in\left[\Delta s, s_\mathsf{{max}}\right]$, where $s_\mathsf{{max}} = (2\Delta{{t}})^{{-1}}$ is the Nyquist frequency and $\Delta s = (N\Delta t)^{{-1}}$ is the smallest frequency that can be resolved.

        $$
            \begin{{align}}
            L_v(s) &= \int_0^\infty C_v(t)\exp{{\left(-st\right)}}dt \\
            D(s) &= -\frac{{L_v(s)\langle(\delta{{}}x)^2\rangle\langle v^2\rangle}}{{L_v(s)\left[s\langle(\delta{{}}x)^2\rangle + s^{{-1}}\langle v^2\rangle\right] -\langle(\delta{{}}x)^2\rangle\langle v^2\rangle}}
            \end{{align}}
        $$

        - The position-dependent diffusivity $D(\langle z \rangle)$ is the limit of $D(s)$ as $s\to 0$. Unfortuantely, $D(s)$ diverges as $s\to 0$ and must be extrapolated from the range of $s$-values where it is well-behaved, instead. `USVTCFAnalysis` fits a fourth-order polynomial to $D(s)$ for $s$-values with a low curvature (deviating from median curvature by less then $5\ \%$). Users should carefully check that this extrapolation is meaningful.

        - Here, we obtain $D = ({vtcf_analysis.diff:.3g} \pm {vtcf_analysis.sem * 1.96:.3g})\ \mathsf{{nm^2\,ps^{{-1}}}}$ at $x = {vtcf_analysis.x:.3g}\ \mathsf{{nm}}$.


        {_plot(vtcf_analysis)}
        """
    )
    return ncv, vtcf_analysis


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numba
    import numpy as np
    import scipy.integrate

    import mvdlib.diffusion
    import mvdlib.plots
    import mvdlib.stats
    import mvdlib.style

    plt.style.use("mvdlib.style.default")
    FIGSIZE = mvdlib.plots.FigSize()
    return FIGSIZE, mo, mvdlib, np, numba, plt, scipy


if __name__ == "__main__":
    app.run()
