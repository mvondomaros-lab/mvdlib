"""Plot types that Matplotlib does not provide directly."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Colormap
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

from mvdlib.style.colors import INK


def plot_contour(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    alpha: float | None = 0.5,
    cmap: str | Colormap | None = None,
    clabels: bool = True,
    grid: bool = False,
    contour_color: ColorType | None = None,
    clabel_color: ColorType | None = None,
    pcolormesh_kwargs: dict | None = None,
    contour_kwargs: dict | None = None,
    clabel_kwargs: dict | None = None,
) -> tuple[QuadMesh, QuadContourSet]:
    """
    Draw a combined color mesh and contour plot.

    Pass the returned mesh to ``Figure.colorbar`` to add a color bar; it
    inherits the mesh transparency.

    Parameters
    ----------
    ax
        Axis to draw on.
    x, y, z
        Coordinates and values passed to ``Axes.pcolormesh`` and ``Axes.contour``.
    alpha
        Color mesh transparency. The default keeps contours legible.
    cmap
        Color mesh colormap. Defaults to the style's ``image.cmap``.
    clabels
        Whether to label the contours with their values.
    grid
        Whether to keep the axis grid. Off by default, since it shows through
        the transparent mesh.
    contour_color
        Contour line color. Defaults to ``INK``.
    clabel_color
        Contour label color. Defaults to ``contour_color``.
    pcolormesh_kwargs
        Additional keyword arguments passed to ``Axes.pcolormesh``.
    contour_kwargs
        Additional keyword arguments passed to ``Axes.contour``.
    clabel_kwargs
        Additional keyword arguments passed to ``ContourSet.clabel``.

    Returns
    -------
    tuple
        Color mesh and contour set.
    """
    pcolormesh_kwargs = {} if pcolormesh_kwargs is None else pcolormesh_kwargs.copy()
    contour_kwargs = {} if contour_kwargs is None else contour_kwargs.copy()
    clabel_kwargs = {} if clabel_kwargs is None else clabel_kwargs.copy()

    ax.grid(grid)

    pcolormesh_kwargs.setdefault("alpha", alpha)
    if cmap is not None:
        pcolormesh_kwargs.setdefault("cmap", cmap)
    contour_color = INK if contour_color is None else contour_color
    clabel_color = contour_color if clabel_color is None else clabel_color
    contour_kwargs.setdefault("colors", contour_color)
    clabel_kwargs.setdefault("colors", clabel_color)

    mesh = ax.pcolormesh(x, y, z, **pcolormesh_kwargs)
    contourset = ax.contour(x, y, z, **contour_kwargs)
    if clabels:
        contourset.clabel(**clabel_kwargs)

    return mesh, contourset


def plot_kde(
    ax: Axes,
    x: ArrayLike,
    *,
    grid: ArrayLike | None = None,
    factor: float | str | None = None,
    fill: bool = False,
    fill_alpha: float = 0.2,
    **kwargs,
) -> Line2D:
    """
    Draw a Gaussian kernel density estimate.

    Parameters
    ----------
    ax
        Axis to draw on.
    x
        Input samples.
    grid
        Evaluation grid. Defaults to 250 points spanning the samples plus
        three kernel bandwidths on either side, so the density decays to zero.
    factor
        Bandwidth factor passed to ``scipy.stats.gaussian_kde`` as
        ``bw_method``.
    fill
        Whether to fill the area under the curve.
    fill_alpha
        Fill transparency.
    **kwargs
        Additional keyword arguments passed to ``Axes.plot``.

    Returns
    -------
    matplotlib.lines.Line2D
        KDE line.
    """
    from scipy.stats import gaussian_kde

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size == 0:
        raise ValueError("x must not be empty")
    kde = gaussian_kde(x, bw_method=factor)
    if grid is None:
        pad = 3.0 * np.sqrt(kde.covariance[0, 0])
        grid = np.linspace(x.min() - pad, x.max() + pad, 250)
    else:
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim != 1:
            raise ValueError("grid must be one-dimensional")
    y = kde(grid)
    (line,) = ax.plot(grid, y, **kwargs)
    if fill:
        ax.fill_between(
            grid,
            y,
            color=line.get_color(),
            alpha=fill_alpha,
            linewidth=0.0,
            zorder=line.get_zorder() - 0.1,
        )
    return line
