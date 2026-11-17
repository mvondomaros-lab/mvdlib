"""Axis labels, panel labels, and tick counts."""

from collections.abc import Iterable
from string import ascii_lowercase

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator


def axis_label(quantity: str, unit: str | None = None) -> str:
    """
    Format an axis label as quantity divided by unit, e.g. ``E / kJ mol⁻¹``.

    Dividing the quantity by its unit (ISO 80000-1) turns the tick labels into
    pure numbers.

    Parameters
    ----------
    quantity
        Quantity symbol in math syntax, e.g. ``"E"`` or ``r"\\rho"``. Typeset
        in italics.
    unit
        Unit in math syntax, e.g. ``"kJ mol^{-1}"``. Typeset upright, with
        spaces rendered as thin spaces. Omit for dimensionless quantities.

    Returns
    -------
    str
        The label, ready to pass to ``Axes.set_xlabel`` or ``Axes.set_ylabel``.
    """
    if unit is None:
        return f"${quantity}$"
    unit = r"\,".join(unit.split())
    return rf"${quantity}\ /\ \mathrm{{{unit}}}$"


def panel_labels(
    axes: Axes | Iterable[Axes],
    labels: Iterable[str] = ascii_lowercase,
    fmt: str = "({})",
) -> None:
    """
    Label the panels of a figure, e.g. ``(a)``, ``(b)``, ...

    The label goes into the left title slot of each axis, before any title
    already there, and therefore shares the bold title style.

    Parameters
    ----------
    axes
        Axes to label, in order. Arrays from ``plt.subplots`` are flattened
        row by row.
    labels
        Labels to use, in order.
    fmt
        Format string applied to each label.
    """
    for ax, label in zip(np.ravel(axes), labels):
        title = ax.get_title(loc="left")
        ax.set_title(f"{fmt.format(label)} {title}".rstrip(), loc="left")


def set_nticks(ax: Axes, n: int = 6, axis: str = "both") -> None:
    """
    Limit the number of major ticks to at most ``n`` at round values.

    Parameters
    ----------
    ax
        Axis to modify. Shared axes are modified together.
    n
        Maximum number of tick intervals.
    axis
        ``"x"``, ``"y"``, or ``"both"``.
    """
    if axis not in ("x", "y", "both"):
        raise ValueError("axis must be 'x', 'y', or 'both'")
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(MaxNLocator(n))
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(MaxNLocator(n))
