from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from typing import Optional, Tuple

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import QuadContourSet
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray


def _register_colormaps():
    """
    Register custom colormaps with matplotlib.

    The following colormaps are registered:
        - "mvdlib.coolwarm": A diverging colormap from cool to warm.s
        - "mvdlib.warmcool": A diverging colormap from warm to cool.
        - "mvdlib.bluered": A blend from blue to red.
        - "mvdlib.redblue": A blend from red to blue.

    This function also imports seaborn, which registers its own colormaps.
    """
    import matplotlib as mpl
    import seaborn as sns

    cmap = sns.diverging_palette(250, 0, l=70, center="dark", as_cmap=True)
    mpl.colormaps.register(cmap, name="mvdlib.coolwarm")

    cmap = sns.diverging_palette(0, 250, l=70, center="dark", as_cmap=True)
    mpl.colormaps.register(cmap, name="mvdlib.warmcool")

    cmap = sns.color_palette("blend:#426ca4,#bb3b5f", as_cmap=True)
    mpl.colormaps.register(cmap, name="mvdlib.bluered")

    cmap = sns.color_palette("blend:#bb3b5f,#426ca4", as_cmap=True)
    mpl.colormaps.register(cmap, name="mvdlib.redblue")


def contourplot(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    *,
    alpha: Optional[float] = 0.5,
    cmap: Optional[str] = "mvdlib.bluered",
    pcolormesh_kwargs: Optional[dict] = None,
    contour_kwargs: Optional[dict] = None,
    clabel_kwargs: Optional[dict] = None,
) -> Tuple[QuadMesh, QuadContourSet]:
    """
    Draw a combined color mesh and contour plot.

    :param ax: The axis to draw on.
    :param x: The x coordinates.
    :param y: The y coordinates.
    :param z: The z coordinates.
    :param alpha: The alpha value for the color mesh.
    :param cmap: A colormap. Default is "mvdlib.bluered".
    :param pcolormesh_kwargs: Dictionary for additional arguments passed to `ax.pcolormesh()`.
    :param contour_kwargs: Dictionary for additional arguments passed to `ax.contour()`.
    :param clabel_kwargs: Dictionary for additional arguments passed to `ax.clabel()`.
    :return: The QuadMesh and QuadContourSet objects returned by `ax.pcolormesh()` and `ax.contour()`, respectively,
    """
    pcolormesh_kwargs = pcolormesh_kwargs or {}
    contour_kwargs = contour_kwargs or {}
    clabel_kwargs = clabel_kwargs or {}

    ax.grid(False)

    pcolormesh_kwargs.setdefault("alpha", alpha)
    pcolormesh_kwargs.setdefault("cmap", cmap)
    contour_kwargs.setdefault("colors", "#ECEFF4")
    clabel_kwargs.setdefault("colors", "#ECEFF4")

    mesh = ax.pcolormesh(x, y, z, **pcolormesh_kwargs)
    contourset = ax.contour(x, y, z, **contour_kwargs)
    contourset.clabel(**clabel_kwargs)

    return mesh, contourset


def kdeplot(
    ax: Axes,
    x: NDArray[np.float64],
    *,
    grid: Optional[NDArray[np.float64]] = None,
    factor: Optional[float] = None,
    **kwargs,
) -> Line2D:
    """
    Draw a Kernel Density Estimate (KDE).

    :param ax: The axis to draw on.
    :param x: The data.
    :param grid: The grid on which the KDE is evaluated. If None, a grid is automatically generated.
    :param factor: The bandwidth factor for the KDE. If None, the default is used.
    :param kwargs: Additional keyword arguments passed to `ax.plot`.
    :return: The Line2D object returned by `ax.plot`.
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(x, bw_method=factor)
    if grid is None:
        grid = np.linspace(x.min() - 0.02 * np.ptp(x), x.max() + 0.02 * np.ptp(x), 250)

    y = kde(grid)
    lines = ax.plot(grid, y, **kwargs)
    # Only one line.
    return lines[0]
