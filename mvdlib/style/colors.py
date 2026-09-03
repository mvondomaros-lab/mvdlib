"""The colour system used by :mod:`mvdlib` plotting styles."""

import seaborn as sns


# Structural ink and semantic roles for line data.
INK = "#30343A"
BASE = "#356FA8"
H1 = "#A03A6C"
H2 = "#C47F00"

LINES = (BASE, H1, H2)

# Seaborn registers these palettes with Matplotlib on import.
SEQ = sns.color_palette("crest", as_cmap=True)
DIV = sns.color_palette("vlag", as_cmap=True)
