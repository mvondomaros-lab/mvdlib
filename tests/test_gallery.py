import importlib.util
import logging
import re
from pathlib import Path

import pytest
from PIL import Image

GALLERY = Path(__file__).parents[1] / "docs" / "gallery.py"


def test_gallery_renders_without_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The style guide figures render, and no font falls back silently."""
    spec = importlib.util.spec_from_file_location("gallery", GALLERY)
    gallery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gallery)

    with caplog.at_level(logging.WARNING, logger="matplotlib"):
        gallery.main(tmp_path)

    assert not caplog.records
    assert {p.stem for p in tmp_path.glob("*.png")} == set(gallery.FIGURES)


@pytest.mark.parametrize("doc", ["README.md", "docs/styleguide.md"])
def test_docs_show_figures_at_print_size(doc: str) -> None:
    """Images display at 96 px per inch, so all figures share one scale."""
    spec = importlib.util.spec_from_file_location("gallery", GALLERY)
    gallery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gallery)

    path = GALLERY.parents[1] / doc
    images = re.findall(r'<img src="([^"]+)"[^>]* width="(\d+)"', path.read_text())
    assert images
    for src, width in images:
        with Image.open(path.parent / src) as image:
            pixels = image.width
        assert int(width) == round(pixels / gallery.DPI * 96), src


def test_site_shows_figures_at_print_size(tmp_path: Path) -> None:
    pytest.importorskip("markdown")
    spec = importlib.util.spec_from_file_location("build_site", GALLERY.parent / "build_site.py")
    build_site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_site)

    build_site.build(tmp_path)

    html = (tmp_path / "index.html").read_text()
    figures = re.findall(r'<div class="figure"><img src="([^"]+)"[^>]* width="(\d+)"', html)
    sources = re.findall(r'<img src="([^"]+)"', (GALLERY.parent / "styleguide.md").read_text())
    assert [src for src, _ in figures] == sources
    for src, width in figures:
        with Image.open(tmp_path / src) as image:
            assert int(width) == round(image.width / build_site.DPI * 96)
    assert (tmp_path / "fonts" / "FiraSans-Regular.ttf").exists()
    assert 'href="gallery.py"' not in html
