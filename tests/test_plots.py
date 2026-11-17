import numpy as np
import pytest
from matplotlib.figure import Figure

import mvdlib.plots as plots
from mvdlib.style import INK


def test_plots_wildcard_exports() -> None:
    namespace: dict[str, object] = {}

    exec("from mvdlib.plots import *", namespace)

    for name in plots.__all__:
        assert namespace[name] is getattr(plots, name)


@pytest.mark.parametrize(
    "quantity, unit, expected",
    [
        ("E", "kJ mol^{-1}", r"$E\ /\ \mathrm{kJ\,mol^{-1}}$"),
        ("t", "ns", r"$t\ /\ \mathrm{ns}$"),
        (r"\rho", None, r"$\rho$"),
    ],
)
def test_axis_label(quantity: str, unit: str | None, expected: str) -> None:
    assert plots.axis_label(quantity, unit) == expected


def test_panel_labels_prefix_existing_titles() -> None:
    axes = Figure().subplots(2, 2)
    axes[0, 1].set_title("system", loc="left")

    plots.panel_labels(axes)

    titles = [ax.get_title(loc="left") for ax in axes.flat]
    assert titles == ["(a)", "(b) system", "(c)", "(d)"]


def test_set_nticks() -> None:
    ax = Figure().subplots()
    xlocator = ax.xaxis.get_major_locator()

    plots.set_nticks(ax, 4, axis="y")

    assert ax.yaxis.get_major_locator()._nbins == 4
    assert ax.xaxis.get_major_locator() is xlocator
    with pytest.raises(ValueError, match="axis must be"):
        plots.set_nticks(ax, axis="z")


def test_plot_contour_smoke() -> None:
    ax = Figure().subplots()
    coordinates = np.linspace(-1.0, 1.0, 8)
    x, y = np.meshgrid(coordinates, coordinates)
    z = x**2 + y**2

    mesh, contourset = plots.plot_contour(ax, x, y, z)

    assert mesh.axes is ax
    assert mesh.get_alpha() == 0.5
    assert contourset.axes is ax
    assert contourset.labelTexts
    assert np.allclose(contourset.get_edgecolor()[0][:3], np.array([0x30, 0x34, 0x3A]) / 255)
    assert INK == "#30343A"


def test_plot_contour_without_labels() -> None:
    ax = Figure().subplots()
    x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))

    _, contourset = plots.plot_contour(ax, x, y, x * y, clabels=False)

    assert not contourset.labelTexts


def test_plot_kde_decays_to_zero() -> None:
    ax = Figure().subplots()
    rng = np.random.default_rng(0)

    line = plots.plot_kde(ax, rng.normal(size=100))

    y = line.get_ydata()
    assert line.axes is ax
    assert y.size == 250
    assert y[0] < 0.02 * y.max() and y[-1] < 0.02 * y.max()


def test_plot_kde_fill() -> None:
    ax = Figure().subplots()

    plots.plot_kde(ax, np.arange(10.0), fill=True)

    assert len(ax.collections) == 1
