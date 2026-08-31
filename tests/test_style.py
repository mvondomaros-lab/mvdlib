from importlib.resources import files

import matplotlib.pyplot as plt


def test_default_style_is_packaged_and_loadable() -> None:
    """The plotting-style-only installation exposes the bundled style sheet."""
    style_path = files("mvdlib.style").joinpath("default.mplstyle")
    with plt.style.context(style_path):
        assert plt.rcParams["axes.grid"]
