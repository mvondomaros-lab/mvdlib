"""
The lab's plotting style and its colour system.

Importing this package registers the bundled fonts and colormaps with
Matplotlib. Activate the style with::

    plt.style.use("mvdlib.style.default")
"""

from mvdlib.style.colors import (
    BASE,
    DIV,
    GRID,
    H1,
    H2,
    H3,
    INK,
    LINES,
    SEQ,
)
from mvdlib.style._fonts import register_fonts

register_fonts()

__all__ = [
    "INK",
    "GRID",
    "BASE",
    "H1",
    "H2",
    "H3",
    "LINES",
    "SEQ",
    "DIV",
]
