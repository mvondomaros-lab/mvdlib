"""
Bundled fonts.

The style uses Fira Sans (SIL Open Font License 1.1, see ``fonts/OFL.txt``).
The font files ship with the package and are registered with Matplotlib on
import, so figures render identically on every machine.
"""

from importlib.resources import files

from matplotlib import font_manager


def register_fonts() -> None:
    """
    Register the bundled fonts with Matplotlib.
    """
    for path in files("mvdlib.style").joinpath("fonts").iterdir():
        if path.name.endswith(".ttf"):
            font_manager.fontManager.addfont(str(path))
