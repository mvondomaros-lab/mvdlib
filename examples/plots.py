import marimo

__generated_with = "0.10.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Matplotlib Style Recommendations""")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import mvdlib.plots

    plt.style.use("mvdlib.style.default")
    FIGSIZE = mvdlib.plots.FigSize()
    return FIGSIZE, mo, mvdlib, np, plt, sns


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt):
    def plot_demo():
        # Recommendation: Separate data processing and plotting.
        np.random.seed(42)
        x = np.linspace(0.0, 1000, 250)
        y = np.cumsum(np.random.randn(x.size))

        # Recommendation: Separate data processing and plotting.
        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))

        ax.plot(x, y, color="C1")
        ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Prefer single column figures.
        Adhere to journal size recommendations.

        **Recommendation:**
        Use math mode for axes labels.

        **Recommendation:**
        Use division to eliminate units.
        Use negative exponents to denote units in the denominator.

        {mo.as_html(plot_demo())}
        """
    )
    return (plot_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt):
    # def color_demo():
    #     np.random.seed(42)
    #     n = 5
    #     x = np.linspace(0.0, 1000, 250)
    #     y = [np.cumsum(np.random.randn(x.size)) for _ in range(n)]

    #     nrows, ncols = 1, 1
    #     fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))
    #     for i in range(n):
    #         ax.plot(x, y[i], color="C1", alpha=0.6)
    #     ax.plot([], [], color="C1", alpha=0.6, label="sample")
    #     ax.plot(x, np.mean(y, axis=0), color="C2", label="mean")
    #     ax.plot(x, np.zeros_like(x), color="C0", label="expected")

    #     ax.legend()
    #     ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
    #     ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
    #     ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    #     return fig


    def color_demo():
        np.random.seed(42)
        n = 4
        x = np.linspace(0.0, 1000, 250)
        y = [np.cumsum(np.random.randn(x.size)) for _ in range(n)]

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))
        for i in range(n):
            ax.plot(x, y[i], color=f"C{i}", label=f"DS{i}")

        ax.legend()
        ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Keep colors to a minimum.

        {mo.as_html(color_demo())}
        """
    )
    return (color_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt):
    def alpha_demo():
        np.random.seed(42)
        n = 10
        x = np.linspace(0.0, 1000, 250)
        y = [np.cumsum(np.random.randn(x.size)) for _ in range(n)]

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))
        for i in range(n):
            ax.plot(x, y[i], color="C1", alpha=0.5)
        ax.plot([], [], color="C1", alpha=0.5, label="sample")
        ax.plot(x, np.mean(y, axis=0), color="C2", label="mean")
        ax.plot(x, np.zeros_like(x), color="C0", label="expected", ls="--")

        ax.legend()
        ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Try to use transparency when plotting many lines.

        {mo.as_html(alpha_demo())}
        """
    )
    return (alpha_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt, sns):
    def shade_demo():
        np.random.seed(42)

        n = 5
        x = np.linspace(0.0, 1000, 250)
        ys = [np.cumsum(np.random.randn(x.size)) for _ in range(n)]
        ys = np.cumsum(ys, axis=0)

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))
        colors = sns.color_palette("blend:C1,C2", n)
        for i in range(n):
            ax.plot(x, ys[i], color=colors[i], label=f"DS{i}")

        ax.legend(ncol=1)
        ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig, ax


    shade_demo()

    mo.md(
        f"""
        **Recommendation:**
        Use sequential colormaps to discriminate data that is connected by some hidden variable, like time. 

        {mo.as_html(shade_demo())}
        """
    )
    return (shade_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt):
    def column_demo():
        np.random.seed(42)

        nrows, ncols = 1, 3
        x = np.linspace(0.0, 1000, 250)
        y = [np.cumsum(np.random.randn(x.size)) for _ in range(ncols)]

        fig, axes = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True, sharey=True
        )
        for i in range(ncols):
            axes[i].plot(x, y[i], color="C1")

            axes[i].set_title(f"DS{i}")
            axes[i].set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")

        axes[0].set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        axes[0].yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Use columns for shared y-axes.
        Use axes titles for designating panels.
        Adhere to journal size recommendations.

        {mo.as_html(column_demo())}
        """
    )
    return (column_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt):
    def row_demo():
        np.random.seed(42)

        nrows, ncols = 3, 1
        x = np.linspace(0.0, 1000, 250)
        y = [np.cumsum(np.random.randn(x.size)) for _ in range(nrows)]

        fig, axes = plt.subplots(
            nrows, ncols, figsize=FIGSIZE(nrows, ncols), sharex=True, sharey=True
        )
        for i in range(nrows):
            axes[i].plot(x, y[i], color="C1")

            axes[i].set_title(f"DS{i}")
            axes[i].set_ylabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")

        axes[-1].set_xlabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        axes[-1].xaxis.set_major_locator(plt.MaxNLocator(6))
        axes[-1].yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Use rows for shared x-axes.

        {mo.as_html(row_demo())}
        """
    )
    return (row_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, mvdlib, np, plt):
    def mesh_demo():
        x = np.linspace(-1.5, 1.5, 250)
        y = np.linspace(-1.5, 1.5, 250)
        x, y = np.meshgrid(x, y)
        z = np.exp(-(x**2) - y**2)

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows * 1.3, ncols))
        mesh, contourset = mvdlib.plots.contourplot(ax, x, y, z)

        ax.set_aspect("equal")
        ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_title(r"$\rho(x, y)dxdy$", loc="center")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Use color meshes and contours to represent a third dimension.

        **Recommendation:**
        Set equal aspect ratios for x and y-axes with the same magnitude.

        {mo.as_html(mesh_demo())}
        """
    )
    return (mesh_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, mvdlib, np, plt):
    def mesh_demo2():
        x = np.linspace(-np.pi, np.pi, 250)
        y = x
        x, y = np.meshgrid(x, y)
        z = np.sin(x) * np.cos(y)

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(1.3 * nrows, ncols))
        mvdlib.plots.contourplot(ax, x, y, z, cmap="mvdlib.coolwarm")

        ax.set_aspect("equal")
        ax.set_xlabel(r"$x\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_ylabel(r"$y\ /\ \mathsf{a\,b^{-1}}$")
        ax.set_title(r"$\rho(x, y)dxdy$", loc="center")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Use diverging color maps when displaying diverging data.

        {mo.as_html(mesh_demo2())}
        """
    )
    return (mesh_demo2,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, np, plt):
    def bar_demo():
        np.random.seed(42)
        N = 5
        classes = [f"Class {i}" for i in range(N)]
        values = np.random.randint(5, 20, N)

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))
        ax.bar(classes, values, alpha=0.5, color="C1")
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.grid(axis="x")
        ax.set_ylabel("count")

        for i in range(N):
            ax.text(
                i,
                values[i] - 0.3,
                f"{values[i]}",
                ha="center",
                va="top",
                color="white",
            )

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Add values when showing bar plots.

        {mo.as_html(bar_demo())}
        """
    )
    return (bar_demo,)


@app.cell(hide_code=True)
def _(FIGSIZE, mo, mvdlib, np, plt):
    def kde_demo():
        np.random.seed(42)
        N = 500
        x = np.random.randn(N)

        nrows, ncols = 1, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=FIGSIZE(nrows, ncols))
        ax.hist(x, alpha=0.5, density=True, color="C1")
        mvdlib.plots.kdeplot(ax, x, color="C1")

        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_xlabel(r"$x\ /\ a\,b^{-1}$")
        ax.set_ylabel(r"$\rho(x)dx$")

        return fig


    mo.md(
        f"""
        **Recommendation:**
        Use kernel density estimation to show the distribution of data.

        {mo.as_html(kde_demo())}
        """
    )
    return (kde_demo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
