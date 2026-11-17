"""
Render the figures of the style guide.

Usage::

    pixi run gallery            # writes docs/figures/*.png
    python docs/gallery.py OUT  # writes OUT/*.png
"""

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.figure import Figure

from mvdlib.plots import (
    axis_label,
    colorbar,
    panel_labels,
    plot_contour,
    plot_kde,
    set_nticks,
    subplots,
)
from mvdlib.style import BASE, DIV, H1, INK, LINES, SEQ

DPI = 200
"""Resolution of the rendered figures. The docs display them at 96 px per inch."""

SQUARE = (1.5, 1.5)
"""Panel size for maps with equal aspect ratio: the default panel height, squared."""

T_LABEL = axis_label("t", "ns")
E_LABEL = axis_label("E", "kJ mol^{-1}")


def random_walks(n: int, size: int = 250, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(n, size)), axis=1)


def lines() -> Figure:
    t = np.linspace(0.0, 1000.0, 250)
    fig, ax = subplots()
    for i, (y, color) in enumerate(zip(random_walks(len(LINES)), LINES)):
        ax.plot(t, y, color=color, label=f"system {i + 1}")
    ax.legend()
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(E_LABEL)
    set_nticks(ax)
    return fig


def highlight() -> Figure:
    n = 10
    t = np.linspace(0.0, 1000.0, 250)
    y = random_walks(n)
    fig, ax = subplots()
    ax.plot(t, y.T, color=BASE, alpha=0.25, linewidth=0.6, zorder=1)
    ax.plot([], [], color=BASE, alpha=0.25, linewidth=0.6, label=f"samples ($n={n}$)")
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), label="expected")
    ax.plot(t, y.mean(axis=0), color=H1, zorder=3, label="mean")
    ax.legend(loc="lower left")
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(E_LABEL)
    set_nticks(ax)
    return fig


def uncertainty() -> Figure:
    t = np.linspace(0.0, 1000.0, 250)
    y = random_walks(50)
    mean, std = y.mean(axis=0), y.std(axis=0)
    fig, ax = subplots()
    ax.fill_between(t, mean - std, mean + std, color=BASE, alpha=0.25, linewidth=0.0, label=r"$\pm 1\sigma$")
    ax.plot(t, mean, color=BASE, label="mean")
    ax.legend(loc="upper left")
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(E_LABEL)
    set_nticks(ax)
    return fig


def sequential() -> Figure:
    n = 5
    t = np.linspace(0.0, 1000.0, 250)
    y = np.cumsum(random_walks(n), axis=0)
    colors = SEQ(np.linspace(0.25, 1.0, n))
    fig, ax = subplots()
    for i in range(n):
        ax.plot(t, y[i], color=colors[i], label=rf"$T = {300 + 10 * i}\ \mathrm{{K}}$")
    ax.legend(loc="lower left", ncols=2)
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(E_LABEL)
    set_nticks(ax)
    return fig


def columns() -> Figure:
    t = np.linspace(0.0, 1000.0, 250)
    y = random_walks(3)
    fig, axes = subplots(1, 3, sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(t, y[i], color=BASE)
        ax.set_title(f"system {i + 1}")
    panel_labels(axes)
    fig.supxlabel(T_LABEL)
    axes[0].set_ylabel(E_LABEL)
    set_nticks(axes[0])
    return fig


def rows() -> Figure:
    t = np.linspace(0.0, 1000.0, 250)
    y = random_walks(3)
    fig, axes = subplots(3, 1, sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(t, y[i], color=BASE)
        ax.set_title(f"system {i + 1}")
    panel_labels(axes)
    fig.supylabel(E_LABEL)
    axes[-1].set_xlabel(T_LABEL)
    set_nticks(axes[-1])
    return fig


def contours() -> Figure:
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 250), np.linspace(-1.5, 1.5, 250))
    z = np.exp(-(x**2) - y**2)
    fig, ax = subplots(panel=SQUARE)
    mesh, _ = plot_contour(ax, x, y, z, cmap=SEQ)
    colorbar(ax, mesh, label=axis_label(r"\rho(x, y)"))
    ax.set_aspect("equal")
    ax.set_xlabel(axis_label("x", "nm"))
    ax.set_ylabel(axis_label("y", "nm"))
    set_nticks(ax)
    return fig


def diverging() -> Figure:
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 250), np.linspace(-np.pi, np.pi, 250))
    z = np.sin(x) * np.cos(y)
    fig, ax = subplots(panel=SQUARE)
    mesh, _ = plot_contour(
        ax, x, y, z, cmap=DIV, pcolormesh_kwargs={"norm": CenteredNorm()},
        contour_kwargs={"levels": [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]},
    )
    colorbar(ax, mesh, label=axis_label(r"\Delta G", "kJ mol^{-1}"))
    ax.set_aspect("equal")
    ax.set_xlabel(axis_label(r"\phi", "rad"))
    ax.set_ylabel(axis_label(r"\psi", "rad"))
    set_nticks(ax)
    return fig


def bars() -> Figure:
    rng = np.random.default_rng(42)
    classes = [f"class {i + 1}" for i in range(5)]
    values = rng.integers(5, 20, len(classes))
    fig, ax = subplots()
    container = ax.bar(classes, values, color=BASE)
    ax.bar_label(container, padding=-11, color="white")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("count")
    set_nticks(ax, axis="y")
    return fig


def density() -> Figure:
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    fig, ax = subplots()
    ax.hist(x, density=True, color=BASE, alpha=0.5)
    plot_kde(ax, x, color=INK)
    ax.set_xlabel(axis_label("x", "nm"))
    ax.set_ylabel(axis_label(r"p(x)", "nm^{-1}"))
    set_nticks(ax)
    return fig


def logscale() -> Figure:
    rng = np.random.default_rng(42)
    t = np.logspace(-2, 3, 30)
    y = t ** 0.8 * np.exp(rng.normal(scale=0.15, size=t.size))
    fig, ax = subplots()
    ax.loglog(t, t ** 0.8, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), label=r"$\propto t^{0.8}$")
    ax.plot(t, y, "o", color=BASE, label="simulation")
    ax.minorticks_on()
    ax.legend()
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(axis_label(r"\langle r^2 \rangle", "nm^2"))
    return fig


FIGURES: dict[str, Callable[[], Figure]] = {
    "lines": lines,
    "highlight": highlight,
    "uncertainty": uncertainty,
    "sequential": sequential,
    "columns": columns,
    "rows": rows,
    "contours": contours,
    "diverging": diverging,
    "bars": bars,
    "density": density,
    "logscale": logscale,
}


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("mvdlib.style.default")
    for name, make in FIGURES.items():
        fig = make()
        fig.savefig(outdir / f"{name}.png", dpi=DPI)
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures")
