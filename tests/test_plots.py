import numpy as np
import pytest
from matplotlib import style
from matplotlib.figure import Figure

import mvdlib.plots as plots


def test_plots_wildcard_exports() -> None:
    namespace: dict[str, object] = {}

    exec("from mvdlib.plots import *", namespace)

    assert namespace["FigSize"] is plots.FigSize
    assert namespace["contourplot"] is plots.contourplot
    assert namespace["kdeplot"] is plots.kdeplot


def test_figsize_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="single_width must be positive"):
        plots.FigSize(single_width=0.0)

    with pytest.raises(ValueError, match="double_width must be finite"):
        plots.FigSize(double_width=np.inf)

    with pytest.raises(ValueError, match="height must be finite"):
        plots.FigSize()(height=np.nan)


def test_packaged_style_loads() -> None:
    style.use("mvdlib.style.default")


def test_contourplot_smoke() -> None:
    ax = Figure().subplots()
    coordinates = np.linspace(-1.0, 1.0, 8)
    x, y = np.meshgrid(coordinates, coordinates)
    z = x**2 + y**2

    mesh, contourset = plots.contourplot(ax, x, y, z)

    assert mesh.axes is ax
    assert contourset.axes is ax
    assert contourset.get_edgecolor()[0].tolist() == [
        48 / 255,
        52 / 255,
        58 / 255,
        1.0,
    ]


def test_kdeplot_smoke() -> None:
    ax = Figure().subplots()
    rng = np.random.default_rng(0)

    line = plots.kdeplot(ax, rng.normal(size=100))

    assert line.axes is ax
    assert line.get_xdata().size == 250
