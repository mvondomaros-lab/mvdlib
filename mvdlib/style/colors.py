"""
The lab's colour system.

Colours have roles rather than meanings:

- ``INK`` draws structure: text, axes, ticks, and reference lines.
- ``BASE`` draws the primary data. Most figures need no other colour.
- ``H1``, ``H2``, and ``H3`` highlight data against the base, in this order.
- ``GRID`` draws grid lines.

The line colours differ in lightness (``H1`` < ``BASE`` < ``H2`` < ``H3``) so
that they remain distinguishable in greyscale and for readers with colour
vision deficiencies. Each has a contrast ratio of at least 3:1 against white.

``SEQ`` and ``DIV`` are a sequential and a diverging colormap, registered with
Matplotlib as ``mvdlib_seq`` and ``mvdlib_div`` (and reversed as
``mvdlib_seq_r`` and ``mvdlib_div_r``).
"""

import matplotlib as mpl
from matplotlib.colors import ListedColormap

from mvdlib.style._luts import DIV_LUT, SEQ_LUT

INK = "#30343A"
GRID = "#E9E9E9"

BASE = "#236DA6"
H1 = "#8E274B"
H2 = "#079281"
H3 = "#BC8C06"

LINES = (BASE, H1, H2, H3)


def _register(name: str, lut: tuple[str, ...]) -> ListedColormap:
    cmap = ListedColormap(lut, name)
    for c in (cmap, cmap.reversed()):
        if c.name not in mpl.colormaps:
            mpl.colormaps.register(c)
    return mpl.colormaps[name]


SEQ = _register("mvdlib_seq", SEQ_LUT)
DIV = _register("mvdlib_div", DIV_LUT)
