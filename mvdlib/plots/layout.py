"""
Figures with fixed panel sizes.

A panel is the visible axes box. :func:`subplots` fixes its size in inches and
grows the figure around it to fit tick labels, axis labels, titles, and color
bars, so that panels are equally large in every figure of a document.
"""

import warnings
from math import isfinite
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.transforms import Bbox

PANEL = (2.0, 1.5)
"""Default panel width and height in inches."""

ACS_SINGLE_COLUMN = 3.25
"""Maximum width of a single-column ACS figure in inches."""

ACS_DOUBLE_COLUMN = 7.0
"""Maximum width of a double-column ACS figure in inches."""

PAD = 3.0 / 72.0
"""Space between neighboring decorations and around the figure, in inches."""

COLORBAR_WIDTH = 0.08
"""Color bar width in inches."""


class PanelLayoutEngine(LayoutEngine):
    """
    Place panels of fixed size and fit the figure around their decorations.

    Runs whenever the figure is drawn or saved. Use :func:`subplots` instead of
    instantiating this class directly.
    """

    _adjust_compatible = False
    _colorbar_gridspec = False

    def __init__(
        self,
        axes: np.ndarray,
        panel: tuple[float, float],
        max_width: float | None = None,
    ) -> None:
        super().__init__()
        self.axes = axes
        self.panel = panel
        self.max_width = max_width
        self.colorbars: dict[Axes, Axes] = {}

    def execute(self, fig: Figure) -> None:
        renderer = fig._get_renderer()
        # Tick placement depends on the panel size, so place the panels at
        # their final size first, then measure and place them again.
        self._place(fig, self._margins(fig, renderer, measure=False))
        self._place(fig, self._margins(fig, renderer, measure=True))

        width = fig.get_size_inches()[0]
        if self.max_width is not None and width > self.max_width + 1e-3:
            warnings.warn(
                f"figure width {width:.2f} in exceeds max_width {self.max_width:.2f} in",
                stacklevel=2,
            )

    def _extents(self, ax: Axes, renderer) -> tuple[float, float, float, float]:
        """Decoration extents (left, right, bottom, top) around a panel, in inches."""
        to_inches = ax.figure.dpi_scale_trans.inverted()
        box = ax.get_position(original=True).transformed(ax.figure.transFigure)
        box = box.transformed(to_inches)
        bboxes = [ax.get_tightbbox(renderer)]
        if ax in self.colorbars:
            bboxes.append(self.colorbars[ax].get_tightbbox(renderer))
        tight = Bbox.union([b for b in bboxes if b is not None]).transformed(to_inches)
        return (
            max(box.x0 - tight.x0, 0.0),
            max(tight.x1 - box.x1, 0.0),
            max(box.y0 - tight.y0, 0.0),
            max(tight.y1 - box.y1, 0.0),
        )

    def _margins(self, fig: Figure, renderer, measure: bool) -> dict:
        nrows, ncols = self.axes.shape
        left = np.zeros((nrows, ncols))
        right = np.zeros((nrows, ncols))
        bottom = np.zeros((nrows, ncols))
        top = np.zeros((nrows, ncols))
        if measure:
            for (i, j), ax in np.ndenumerate(self.axes):
                left[i, j], right[i, j], bottom[i, j], top[i, j] = self._extents(ax, renderer)

        supx, supy, title = (
            _text_size(fig, t, renderer)
            for t in (fig._supxlabel, fig._supylabel, fig._suptitle)
        )
        return {
            "left": left[:, 0].max() + PAD + (supy[0] + PAD if supy else 0.0),
            "right": right[:, -1].max() + PAD,
            "bottom": bottom[-1, :].max() + PAD + (supx[1] + PAD if supx else 0.0),
            "top": top[0, :].max() + PAD + (title[1] + PAD if title else 0.0),
            "wspace": [right[:, j].max() + left[:, j + 1].max() + PAD for j in range(ncols - 1)],
            "hspace": [bottom[i, :].max() + top[i + 1, :].max() + PAD for i in range(nrows - 1)],
        }

    def _place(self, fig: Figure, m: dict) -> None:
        nrows, ncols = self.axes.shape
        w, h = self.panel
        width = m["left"] + ncols * w + sum(m["wspace"]) + m["right"]
        height = m["bottom"] + nrows * h + sum(m["hspace"]) + m["top"]
        if not np.allclose(fig.get_size_inches(), (width, height)):
            fig.set_size_inches(width, height, forward=True)

        x0 = m["left"] + np.concatenate([[0.0], np.cumsum(w + np.asarray(m["wspace"]))])
        y1 = height - m["top"] - np.concatenate([[0.0], np.cumsum(h + np.asarray(m["hspace"]))])
        for (i, j), ax in np.ndenumerate(self.axes):
            ax.set_position([x0[j] / width, (y1[i] - h) / height, w / width, h / height])
            if ax in self.colorbars:
                cax = self.colorbars[ax]
                x = (x0[j] + w + PAD) / width
                cax.set_position([x, (y1[i] - h) / height, COLORBAR_WIDTH / width, h / height])

        xmid = (m["left"] + (width - m["right"])) / 2.0 / width
        ymid = (m["bottom"] + (height - m["top"])) / 2.0 / height
        if fig._supxlabel is not None:
            fig._supxlabel.set(x=xmid, y=PAD / height, ha="center", va="bottom")
        if fig._supylabel is not None:
            fig._supylabel.set(x=PAD / width, y=ymid, ha="left", va="center")
        if fig._suptitle is not None:
            fig._suptitle.set(x=xmid, y=1.0 - PAD / height, ha="center", va="top")


def _text_size(fig: Figure, text, renderer) -> tuple[float, float] | None:
    if text is None or not text.get_visible() or not text.get_text():
        return None
    bbox = text.get_window_extent(renderer)
    return bbox.width / fig.dpi, bbox.height / fig.dpi


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    panel: tuple[float, float] = PANEL,
    max_width: float | None = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
    squeeze: bool = True,
    subplot_kw: dict | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """
    Create a figure with a grid of panels of fixed size.

    Unlike ``plt.subplots``, the panel size is fixed and the figure size
    follows from it: the figure grows to fit tick labels, axis labels, titles,
    figure-level labels (``fig.supxlabel``, ``fig.supylabel``,
    ``fig.suptitle``), and color bars added with :func:`colorbar`. Use the same
    ``panel`` for every figure of a document to make all panels equally large.

    The layout is recomputed whenever the figure is drawn or saved.

    Parameters
    ----------
    nrows, ncols
        Number of panel rows and columns.
    panel
        Panel width and height in inches, measured along the axis lines.
    max_width
        Maximum figure width in inches, e.g. ``ACS_SINGLE_COLUMN``. A wider
        figure raises a warning when drawn.
    sharex, sharey, squeeze, subplot_kw
        As for ``plt.subplots``.

    Returns
    -------
    fig
        The figure.
    ax
        A single axis, or an array of axes, as for ``plt.subplots``.
    """
    panel = tuple(_validate_dimension(name, v) for name, v in zip(("panel width", "panel height"), panel))
    fig = plt.figure(layout="none")
    axes = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=False, subplot_kw=subplot_kw)
    for ax in axes.flat:
        # Keep the panel size when the aspect ratio is fixed. Matplotlib
        # forbids this for axes shared in both directions.
        if not (sharex and sharey):
            ax.set_adjustable("datalim")
    fig.set_layout_engine(PanelLayoutEngine(axes, panel, max_width))
    fig.get_layout_engine().execute(fig)
    if squeeze:
        return fig, axes.item() if axes.size == 1 else axes.squeeze()
    return fig, axes


def colorbar(ax: Axes, mappable: ScalarMappable, **kwargs) -> Colorbar:
    """
    Add a color bar next to a panel without shrinking it.

    In a figure from :func:`subplots`, the color bar has a fixed width and
    the figure grows to fit it. In other figures, this is
    ``Figure.colorbar(mappable, ax=ax)``.

    Parameters
    ----------
    ax
        Panel to attach the color bar to.
    mappable
        The image, mesh, or contour set the color bar describes.
    **kwargs
        Additional keyword arguments passed to ``Figure.colorbar``.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The color bar.
    """
    fig = ax.figure
    engine = fig.get_layout_engine()
    if not isinstance(engine, PanelLayoutEngine):
        return fig.colorbar(mappable, ax=ax, **kwargs)
    cax = fig.add_axes((0.0, 0.0, 0.0, 0.0))
    engine.colorbars[ax] = cax
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    engine.execute(fig)
    return cbar



def _validate_dimension(name: str, value: float) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value
