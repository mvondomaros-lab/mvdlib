import re
from importlib.resources import files
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.colors import to_rgb

import mvdlib.style.colors as colors
from mvdlib.style import BASE, DIV, GRID, H1, H2, H3, INK, LINES, SEQ

STYLE = files("mvdlib.style").joinpath("default.mplstyle")

# Colour vision deficiency simulation matrices for linear sRGB at full
# severity (Machado, Oliveira, and Fernandes, 2009).
CVD = {
    "deuteranopia": [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    "protanopia": [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    "tritanopia": [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ],
}
SRGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
WHITE = np.array([0.9505, 1.0, 1.089])


def linear(color: str) -> np.ndarray:
    c = np.array(to_rgb(color))
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab(rgb: np.ndarray) -> np.ndarray:
    xyz = SRGB_TO_XYZ @ np.clip(rgb, 0.0, 1.0) / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def min_distance(simulation: np.ndarray) -> float:
    points = [lab(simulation @ linear(c)) for c in LINES]
    return min(np.linalg.norm(a - b) for a, b in combinations(points, 2))


def test_style_loads() -> None:
    with plt.style.context("mvdlib.style.default"):
        assert tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"]) == LINES
        assert plt.rcParams["image.cmap"] == "mvdlib_seq"


def test_style_colours_match_colour_system() -> None:
    palette = {c.upper() for c in (INK, GRID, *LINES)}
    used = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}", STYLE.read_text())}
    assert used == palette


def test_line_colours_are_ordered_by_lightness() -> None:
    lightness = [lab(linear(c))[0] for c in (H1, BASE, H2, H3)]
    assert np.all(np.diff(lightness) >= 6.0)


@pytest.mark.parametrize("color", [BASE, H1, H2, H3, INK])
def test_colours_contrast_with_white(color: str) -> None:
    luminance = (SRGB_TO_XYZ @ linear(color))[1]
    assert 1.05 / (luminance + 0.05) >= 3.0


@pytest.mark.parametrize(
    "deficiency, threshold",
    [("normal", 40.0), ("deuteranopia", 15.0), ("protanopia", 15.0), ("tritanopia", 10.0)],
)
def test_line_colours_remain_distinguishable(deficiency: str, threshold: float) -> None:
    simulation = np.eye(3) if deficiency == "normal" else np.array(CVD[deficiency])
    assert min_distance(simulation) >= threshold


def test_colormaps_are_registered() -> None:
    assert SEQ == mpl.colormaps["mvdlib_seq"]
    assert DIV == mpl.colormaps["mvdlib_div"]
    assert mpl.colormaps["mvdlib_seq_r"].N == SEQ.N == 256
    assert mpl.colormaps["mvdlib_div_r"].N == DIV.N == 256


def test_colormap_registration_is_idempotent() -> None:
    assert colors._register("mvdlib_seq", ("#000000", "#FFFFFF")) == SEQ


def test_bundled_fonts_are_registered() -> None:
    bundled = files("mvdlib.style").joinpath("fonts")
    names = {f.fname for f in font_manager.fontManager.ttflist}
    for path in bundled.iterdir():
        if path.name.endswith(".ttf"):
            assert str(path) in names


def test_style_resolves_text_and_math_to_fira_sans() -> None:
    with plt.style.context("mvdlib.style.default"):
        for prop in (
            font_manager.FontProperties(family=["sans-serif"]),
            plt.rcParams["mathtext.rm"],
            plt.rcParams["mathtext.it"],
        ):
            path = font_manager.findfont(prop, fallback_to_default=False)
            assert "FiraSans" in path.replace(" ", "")


def test_style_exports_editable_text() -> None:
    with plt.style.context("mvdlib.style.default"):
        assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["ps.fonttype"] == 42
        assert plt.rcParams["svg.fonttype"] == "none"
        assert plt.rcParams["savefig.dpi"] >= 300
