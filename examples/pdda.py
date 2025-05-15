import marimo

__generated_with = "0.11.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(FIGSIZE, mo, mvdlib, np, plt, scipy):
    NSTEPS = 5000000
    DT = 0.001

    MASS = 18.0
    DIFF = 0.003
    TEMP = 300.0

    KT = scipy.constants.gas_constant * TEMP / 1000.0
    FRICTION = KT / DIFF


    def plot_sample_free():
        t = np.arange(NSTEPS) * DT
        x, _ = mvdlib.diffusion.ld(
            friction=FRICTION,
            nsteps=NSTEPS,
            dt=DT,
            mass=MASS,
            kt=KT,
            x0=0.0,
            rng=np.random.default_rng(42),
        )

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(t, x, color="C1")
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(r"$x\ /\ \mathsf{nm}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title("Trajectory Sample")

        return fig


    mo.md(
        rf"""

        # Langevin Dynamics

        ## Free Particle

        We start by simulating Langevin dynamics without an external force. We consider a particle with mass $m = {MASS}\ \mathsf{{u}}$, diffusing with a constant diffusivity of $D = {DIFF}\ \mathsf{{nm^2\ ps^{{-1}}}}$ at a temperature $T = {TEMP}\ \mathsf{{K}}$. This mimicks water. Here is a trajectory sample simulated with a time step of $\Delta{{}}t = {DT}\ \mathsf{{ps}}$.

        {mo.as_html(plot_sample_free())}
        """
    )
    return DIFF, DT, FRICTION, KT, MASS, NSTEPS, TEMP, plot_sample_free


@app.cell(hide_code=True)
def _(DIFF, DT, FIGSIZE, FRICTION, KT, MASS, NSTEPS, mo, mvdlib, np, plt):
    NSAMPLES = 50


    def plot_msd_free():
        msds = []
        rng = np.random.default_rng(42)
        for n in range(NSAMPLES):
            x, _ = mvdlib.diffusion.ld(
                friction=FRICTION,
                nsteps=NSTEPS,
                dt=DT,
                mass=MASS,
                kt=KT,
                x0=0.0,
                rng=rng,
            )
            msd = mvdlib.diffusion.msd(x, maxsteps=NSTEPS // 50)
            msds.append(msd)

        msds = np.transpose(msds)
        mean_msd = np.mean(msds, axis=1)
        t = np.arange(mean_msd.size) * DT

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(t, msds, color="C1", alpha=0.1)
        ax.plot(t, mean_msd, color="C2", label="Mean")
        ax.plot(t, 2.0 * DIFF * t, color="C0", ls="--", label="$2Dt$")
        ax.legend()
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(
            r"$\left<{\left[x(t)-x(0)\right]^2}\right>\ /\ \mathsf{nm}^2$"
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title("Mean Squared Displacement")

        return fig


    mo.md(
        rf"""

        ### MSD Analysis

        For a free particle, we can recover the diffusivity, by computing the mean squared displacement, which approaches $2Dt$ in the limit of long times.

        $$
        \braket{{\left[x(t)-x(0)\right]^2}} \to 2Dt
        $$

        {mo.as_html(plot_msd_free())}
        """
    )
    return NSAMPLES, plot_msd_free


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, numba, plt):
    FORCE_CONSTANT = 1000.0
    POTENTIAL_WIDTH = 1.5


    @numba.jit(nopython=False, fastmath=True)
    def force_bounded(x, k=FORCE_CONSTANT, w=POTENTIAL_WIDTH):
        if x < 0.0:
            return -k * x
        elif x < w:
            return 0.0
        else:
            return -k * (x - w)


    def plot_force_potential_bounded():
        x = np.linspace(-0.05 * POTENTIAL_WIDTH, 1.05 * POTENTIAL_WIDTH, 250)
        force = [force_bounded(x) for x in x]
        potential = -np.cumsum(force) * (x[1] - x[0])
        potential -= potential.min()

        nrows, ncols = 1, 2
        fig, axes = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True
        )

        axes[0].plot(x, potential, color="C1")
        axes[1].plot(x, force, color="C1")

        for ax in axes:
            ax.set_xlabel(r"$x\ /\ \mathsf{nm}$")
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        axes[0].set_ylabel(r"$V(x)\ /\ \mathsf{kJ\ mol^{-1}}$")
        axes[0].set_title("Potential")

        axes[1].set_ylabel(r"$F(x)\ /\ \mathsf{kJ\ mol^{-1}\ nm^{-1}}$")
        axes[1].set_title("Force")

        return fig


    mo.md(
        rf"""

        ## Bounded Particle

        Next, we simulate Langevin dynamics in a flat, harmonically bounded potential.

        {mo.as_html(plot_force_potential_bounded())}
        """
    )
    return (
        FORCE_CONSTANT,
        POTENTIAL_WIDTH,
        force_bounded,
        plot_force_potential_bounded,
    )


@app.cell(hide_code=True)
def _(
    DT,
    FIGSIZE,
    FRICTION,
    KT,
    MASS,
    NSTEPS,
    POTENTIAL_WIDTH,
    force_bounded,
    mo,
    mvdlib,
    np,
    plt,
):
    def plot_sample_bounded():
        t = np.arange(NSTEPS) * DT
        x, _ = mvdlib.diffusion.ld(
            force=force_bounded,
            friction=FRICTION,
            nsteps=NSTEPS,
            dt=DT,
            mass=MASS,
            kt=KT,
            x0=0.5 * POTENTIAL_WIDTH,
            rng=np.random.default_rng(42),
        )

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(t, x, color="C1")
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(r"$x\ /\ \mathsf{nm}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title("Trajectory Sample")

        return fig


    mo.md(
        rf"""
        {mo.as_html(plot_sample_bounded())}
        """
    )
    return (plot_sample_bounded,)


@app.cell(hide_code=True)
def _(
    DIFF,
    DT,
    FIGSIZE,
    FRICTION,
    KT,
    MASS,
    NSAMPLES,
    NSTEPS,
    POTENTIAL_WIDTH,
    force_bounded,
    mo,
    mvdlib,
    np,
    plt,
):
    def plot_msd_bounded():
        msds = []
        rng = np.random.default_rng(42)
        for n in range(NSAMPLES):
            x, _ = mvdlib.diffusion.ld(
                force=force_bounded,
                friction=FRICTION,
                nsteps=NSTEPS,
                dt=DT,
                mass=MASS,
                kt=KT,
                x0=0.5 * POTENTIAL_WIDTH,
                rng=rng,
            )
            msd = mvdlib.diffusion.msd(x, maxsteps=NSTEPS // 50)
            msds.append(msd)

        msds = np.transpose(msds)
        mean_msd = np.mean(msds, axis=1)
        t = np.arange(mean_msd.size) * DT

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(t, msds, color="C1", alpha=0.1)
        ax.plot(t, mean_msd, color="C2", label="Mean")
        ax.plot(t, 2.0 * DIFF * t, color="C0", ls="--", label="$2Dt$")
        ax.legend()
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(
            r"$\left<{\left[x(t)-x(0)\right]^2}\right>\ /\ \mathsf{nm}^2$"
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title("Mean Squared Displacement")

        return fig


    mo.md(
        rf"""

        ### MSD Analysis

        The mean squared displacement does not help us anymore.

        {mo.as_html(plot_msd_bounded())}
        """
    )
    return (plot_msd_bounded,)


@app.cell(hide_code=True)
def _(
    DT,
    FIGSIZE,
    FRICTION,
    KT,
    MASS,
    NSTEPS,
    POTENTIAL_WIDTH,
    force_bounded,
    mo,
    mvdlib,
    np,
    plt,
):
    def plot_survival_bounded():
        x, _ = mvdlib.diffusion.ld(
            force=force_bounded,
            friction=FRICTION,
            nsteps=NSTEPS,
            dt=DT,
            mass=MASS,
            kt=KT,
            x0=0.5 * POTENTIAL_WIDTH,
            rng=np.random.default_rng(42),
        )

        survival = mvdlib.diffusion.interval_survival_probability(
            x, xmin=0.0, xmax=POTENTIAL_WIDTH
        )
        residence = mvdlib.diffusion.interval_residence_time(
            x, xmin=0.0, xmax=POTENTIAL_WIDTH, dt=DT
        )
        diff = mvdlib.diffusion.interval_diffusivity(
            x, xmin=0.0, xmax=POTENTIAL_WIDTH, dt=DT
        )
        t = np.arange(survival.size) * DT

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(t, survival, color="C1")
        ax.text(
            1.0, 0.25, rf"$\tau = {residence:.2g}\ \mathsf{{ps}}$", va="center"
        )
        ax.text(
            1.0,
            0.15,
            rf"$D = {diff:.2g}\ \mathsf{{nm^2\ ps^{{-1}}}}$",
            va="center",
        )
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(r"$P(t)$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title("Survival Probability")

        return fig


    mo.md(
        rf"""

        ### Residence Time Analysis

        We define a domain $\Omega = [x_\mathsf{{min}}, x_\mathsf{{max}})$ and compute the survival probability $P(t)$, that is, the probability of the particle staying within $\Omega$, given that it was within the domain initially.
        We may also compute the residence time

        $$
        \tau = \int_0^\infty P(t) dt\,.
        $$

        From this, we obtain a diffusivity $D$.

        $$
        D = \frac{{\left(x_\mathsf{{max}}-x_\mathsf{{min}}\right)^2}}{{12\tau}}
        $$

        For a bounded particle in a homogeneous medium, we may define $\Omega$ as the region of the potential well.

        {mo.as_html(plot_survival_bounded())}
        """
    )
    return (plot_survival_bounded,)


@app.cell(hide_code=True)
def _(
    DIFF,
    DT,
    FIGSIZE,
    FRICTION,
    KT,
    MASS,
    NSAMPLES,
    NSTEPS,
    POTENTIAL_WIDTH,
    force_bounded,
    mo,
    mvdlib,
    np,
    plt,
):
    def plot_diffusivity_bounded():
        rng = np.random.default_rng(42)
        bin_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5])
        diffs = []
        for bin_size in bin_sizes:
            diffs.append([])
            for n in range(NSAMPLES):
                x, _ = mvdlib.diffusion.ld(
                    force=force_bounded,
                    friction=FRICTION,
                    nsteps=NSTEPS,
                    dt=DT,
                    mass=MASS,
                    kt=KT,
                    x0=0.5 * POTENTIAL_WIDTH,
                    rng=rng,
                )
                diff = mvdlib.diffusion.interval_diffusivity(
                    x,
                    xmin=0.5 * (POTENTIAL_WIDTH - bin_size),
                    xmax=0.5 * (POTENTIAL_WIDTH + bin_size),
                    dt=DT,
                )
                diffs[-1].append(diff)

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.violinplot(diffs, showmeans=True, showextrema=True)
        ax.set_xticks(
            np.arange(len(bin_sizes)) + 1,
            labels=[f"{bs:.2g}" for bs in bin_sizes],
            rotation=45,
        )
        ax.set_xlabel(r"$\Delta{}x\ /\ \mathsf{nm}$")
        ax.set_ylabel(r"$D\ /\ \mathsf{nm^2\ ps^{-1}}$")
        ax.axhline(DIFF, color="C2", label="Input")
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.legend()
        ax.set_title("Diffusivities")
        return fig


    mo.md(
        rf"""

        The results of the residence time analysis depend on the domain size $\Delta{{}}x = x_\mathsf{{max}} - x_\mathsf{{min}}$.

        {mo.as_html(plot_diffusivity_bounded())}
        """
    )
    return (plot_diffusivity_bounded,)


@app.cell(hide_code=True)
def _(FIGSIZE, FRICTION, KT, POTENTIAL_WIDTH, mo, np, numba, plt):
    FRICTION_SCALE = 5000.0
    FRICTION_WIDTH = 0.04


    @numba.jit(nopython=False, fastmath=True)
    def friction_peak(
        x,
        xc=0.5 * POTENTIAL_WIDTH,
        f0=FRICTION,
        df=FRICTION_SCALE,
        sigma=FRICTION_WIDTH,
    ):
        return f0 + df * np.exp(-((x - xc) ** 2) / sigma)


    def plot_diff_profile():
        x = np.linspace(-0.05 * POTENTIAL_WIDTH, 1.05 * POTENTIAL_WIDTH, 250)
        friction = np.array([friction_peak(x) for x in x])
        diff = KT / friction

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True
        )

        ax.plot(x, diff, color="C1")

        ax.set_xlabel(r"$x\ /\ \mathsf{nm}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        ax.set_ylabel(r"$D(x)\ /\ \mathsf{nm^2\ ps^{-1}}$")
        ax.set_title("Diffusivity")

        return fig


    mo.md(
        rf"""
        ## Inhomogeneous Diffusivity

        We continue to study a particle bounded by harmonic walls, but this time, we have a position-dependent diffusivity $D(x)$. Note that the dynamical bottleneck leads to uneven sampling.

        {mo.as_html(plot_diff_profile())}
        """
    )
    return FRICTION_SCALE, FRICTION_WIDTH, friction_peak, plot_diff_profile


@app.cell(hide_code=True)
def _(
    DT,
    FIGSIZE,
    KT,
    MASS,
    NSTEPS,
    force_bounded,
    friction_peak,
    mo,
    mvdlib,
    np,
    plt,
):
    def plot_sample_peak():
        t = np.arange(NSTEPS) * DT
        x, _ = mvdlib.diffusion.ld(
            force=force_bounded,
            friction=friction_peak,
            nsteps=NSTEPS,
            dt=DT,
            mass=MASS,
            kt=KT,
            x0=0.0,
            rng=np.random.default_rng(42),
        )

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(t, x, color="C1")
        ax.set_xlabel(r"$t\ /\ \mathsf{ps}$")
        ax.set_ylabel(r"$x\ /\ \mathsf{nm}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title("Trajectory Sample")

        return fig


    mo.md(
        rf"""
        {mo.as_html(plot_sample_peak())}
        """
    )
    return (plot_sample_peak,)


@app.cell(hide_code=True)
def _(
    DT,
    FIGSIZE,
    KT,
    MASS,
    NSAMPLES,
    NSTEPS,
    POTENTIAL_WIDTH,
    force_bounded,
    friction_peak,
    mo,
    mvdlib,
    np,
    plt,
):
    def plot_diff_profile_simulated():
        bins = np.linspace(0.0, POTENTIAL_WIDTH, 4, endpoint=True)
        diffs = []
        rng = np.random.default_rng(42)
        for n in range(NSAMPLES):
            diffs.append([])
            for i in range(bins.size - 1):
                xmin, xmax = bins[i], bins[i + 1]
                x, _ = mvdlib.diffusion.ld(
                    force=force_bounded,
                    friction=friction_peak,
                    nsteps=NSTEPS,
                    dt=DT,
                    mass=MASS,
                    kt=KT,
                    x0=0.0,
                    rng=rng,
                )
                diff = mvdlib.diffusion.interval_diffusivity(
                    x,
                    xmin=xmin,
                    xmax=xmax,
                    dt=DT,
                )
                diffs[-1].append(diff)

        x_ref = np.linspace(-0.05 * POTENTIAL_WIDTH, 1.05 * POTENTIAL_WIDTH, 250)
        friction_ref = np.array([friction_peak(x) for x in x_ref])
        diff_ref = KT / friction_ref

        diffs = np.array(diffs)
        diff_x = 0.5 * (bins[1:] + bins[:-1])
        diff_mean = np.mean(diffs, axis=0)
        diff_std = np.std(diffs, ddof=1)

        nrows = 1
        ncols = 1
        fig, ax = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True
        )

        ax.plot(x_ref, diff_ref, color="C0", ls="--", label="Input")

        ax.errorbar(
            x=diff_x, y=diff_mean, yerr=1.95 * diff_std, color="C1", marker="."
        )

        ax.set_xlabel(r"$x\ /\ \mathsf{nm}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        ax.set_ylabel(r"$D(x)\ /\ \mathsf{nm^2\ ps^{-1}}$")
        ax.set_title("Diffusivity")

        return fig


    mo.md(
        rf"""
        ### Residence Time Analysis

        Here, we chose three non-overlapping bins. The constant diffusivity at the domain edges is somewhat underestimated, because the assumption of a uniform diffusivity within each bin is not strictly valid.

        {mo.as_html(plot_diff_profile_simulated())}
        """
    )
    return (plot_diff_profile_simulated,)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numba
    import numpy as np
    import scipy.constants

    import mvdlib.diffusion
    import mvdlib.plots

    plt.style.use("mvdlib.style.default")
    FIGSIZE = mvdlib.plots.FigSize(w=4, h=3)
    return FIGSIZE, mo, mvdlib, np, numba, plt, scipy


if __name__ == "__main__":
    app.run()
