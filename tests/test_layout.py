import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from mvdlib.plots import ACS_DOUBLE_COLUMN, PANEL, axis_label, colorbar, subplots


@pytest.fixture(autouse=True)
def style():
    with plt.style.context("mvdlib.style.default"):
        yield
    plt.close("all")


def panel_sizes(fig: Figure) -> list[tuple[float, float]]:
    width, height = fig.get_size_inches()
    return [
        (ax.get_position().width * width, ax.get_position().height * height)
        for ax in fig.axes
    ]


def assert_fits(fig: Figure) -> None:
    width, height = fig.get_size_inches()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    assert bbox.x0 >= -1e-6 and bbox.y0 >= -1e-6
    assert bbox.x1 <= width + 1e-6 and bbox.y1 <= height + 1e-6


def save(fig: Figure) -> None:
    from io import BytesIO

    fig.savefig(BytesIO(), format="png")


def labelled(ax, ylabel: str = axis_label("E", "kJ mol^{-1}")) -> None:
    ax.plot(np.linspace(0.0, 1000.0, 10), np.linspace(-30.0, 30.0, 10))
    ax.set_xlabel(axis_label("t", "ns"))
    ax.set_ylabel(ylabel)


def test_panels_are_equal_across_layouts() -> None:
    fig1, ax = subplots()
    labelled(ax)
    fig3, axes = subplots(2, 3, sharey=True)
    for a in axes.flat:
        labelled(a)
        a.set_title("title")
    for fig in (fig1, fig3):
        save(fig)
        assert np.allclose(panel_sizes(fig), PANEL)
        assert_fits(fig)


def test_figure_grows_with_decorations() -> None:
    fig_short, ax = subplots()
    labelled(ax, "y")
    fig_long, ax = subplots()
    labelled(ax, "a much longer label")
    ax.set_yticks([0.0], ["a long tick label"])
    save(fig_short)
    save(fig_long)

    assert fig_long.get_size_inches()[0] > fig_short.get_size_inches()[0]
    assert np.allclose(panel_sizes(fig_long), PANEL)
    assert_fits(fig_long)


def test_layout_ignores_constrained_layout_rc() -> None:
    assert plt.rcParams["figure.constrained_layout.use"]
    fig, ax = subplots(panel=(1.0, 1.0))
    labelled(ax)
    save(fig)

    assert np.allclose(panel_sizes(fig), (1.0, 1.0))


def test_colorbar_keeps_panel_size() -> None:
    fig, ax = subplots(panel=(1.5, 1.5))
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 5), np.linspace(-1.0, 1.0, 5))
    mesh = ax.pcolormesh(x, y, x * y)
    colorbar(ax, mesh, label="z")
    ax.set_aspect("equal")
    save(fig)

    width, height = fig.get_size_inches()
    panel, cbar = fig.axes
    assert np.allclose(panel_sizes(fig)[0], (1.5, 1.5))
    assert cbar.get_position().x0 > panel.get_position().x1
    assert_fits(fig)


def test_colorbar_falls_back_outside_panel_layout() -> None:
    fig = plt.figure()
    ax = fig.subplots()
    mesh = ax.pcolormesh(np.arange(9.0).reshape(3, 3))

    cbar = colorbar(ax, mesh)

    assert cbar.ax in fig.axes


def test_figure_labels_are_centered_on_panels() -> None:
    fig, axes = subplots(1, 3, sharey=True)
    for a in axes:
        labelled(a)
        a.set_xlabel("")
    fig.supxlabel("shared")
    save(fig)

    width, _ = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()
    label = fig._supxlabel.get_window_extent(renderer)
    left = axes[0].get_position().x0 * width
    right = axes[-1].get_position().x1 * width
    assert label.x0 / fig.dpi + label.width / fig.dpi / 2 == pytest.approx((left + right) / 2, abs=0.01)
    assert_fits(fig)


def test_max_width_warns() -> None:
    with pytest.warns(UserWarning, match="exceeds max_width"):
        fig, axes = subplots(1, 4, max_width=ACS_DOUBLE_COLUMN)
        for a in axes:
            labelled(a)
        save(fig)


def test_max_width_is_silent_when_respected() -> None:
    fig, axes = subplots(1, 3, sharey=True, max_width=ACS_DOUBLE_COLUMN)
    for a in axes:
        labelled(a, ylabel="")
    axes[0].set_ylabel(axis_label("E", "kJ mol^{-1}"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save(fig)


def test_squeeze() -> None:
    _, ax = subplots()
    _, row = subplots(1, 3)
    _, grid = subplots(1, 1, squeeze=False)

    assert not isinstance(ax, np.ndarray)
    assert row.shape == (3,)
    assert grid.shape == (1, 1)


def test_rejects_invalid_panel() -> None:
    with pytest.raises(ValueError, match="panel width must be positive"):
        subplots(panel=(0.0, 1.0))
