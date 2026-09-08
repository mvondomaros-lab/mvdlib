from importlib.resources import files

import matplotlib.pyplot as plt

from mvdlib.style import (
    BASE,
    DIV,
    H1,
    H2,
    H3,
    LINES,
    SEQ,
)


def test_default_style_is_packaged_and_loadable() -> None:
    """The plotting-style-only installation exposes the bundled style sheet."""
    style_path = files("mvdlib.style").joinpath("default.mplstyle")
    with plt.style.context(style_path):
        assert tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"]) == LINES
        assert plt.rcParams["axes.grid"]
        assert plt.rcParams["image.cmap"] == "crest"


def test_colour_system_exposes_seaborn_colormaps() -> None:
    assert LINES == (BASE, H1, H2, H3)
    assert SEQ.name == "crest"
    assert DIV.name == "vlag"
