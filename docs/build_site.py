"""
Build the style guide website.

Converts ``docs/styleguide.md`` into a single HTML page that shows every figure
at its printed size: 96 CSS pixels per inch, never scaled down to the page
width. Figures wider than the window scroll horizontally instead, so panels
compare across single- and multi-panel figures.

Usage::

    pixi run site            # writes site/
    python docs/build_site.py OUT
"""

import re
import shutil
import sys
from pathlib import Path

import markdown
from PIL import Image

DOCS = Path(__file__).parent
FONTS = DOCS.parent / "mvdlib" / "style" / "fonts"

sys.path.insert(0, str(DOCS))
from gallery import DPI  # noqa: E402

CSS = """
@font-face { font-family: "Fira Sans"; src: url("fonts/FiraSans-Regular.ttf"); }
@font-face { font-family: "Fira Sans"; src: url("fonts/FiraSans-Italic.ttf"); font-style: italic; }
@font-face { font-family: "Fira Sans"; src: url("fonts/FiraSans-Bold.ttf"); font-weight: bold; }
@font-face {
  font-family: "Fira Sans"; src: url("fonts/FiraSans-BoldItalic.ttf");
  font-weight: bold; font-style: italic;
}
:root {
  --ink: #30343A; --muted: #6B7078; --rule: #E9E9E9; --code: #F5F6F7;
  --accent: #236DA6; --page: #FFFFFF;
}
* { box-sizing: border-box; }
html { background: var(--page); }
body {
  margin: 0 auto; max-width: 46rem; padding: 2rem 16px 4rem;
  font: 17px/1.6 "Fira Sans", system-ui, sans-serif; color: var(--ink);
  background: var(--page);
}
h1 { font-size: 2rem; margin: 0 0 1rem; }
h2 { font-size: 1.3rem; margin: 2.5rem 0 0.75rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
a { color: var(--accent); }
code { font: 0.85em/1.4 ui-monospace, "SF Mono", Menlo, monospace; background: var(--code);
       padding: 0.1em 0.3em; border-radius: 3px; }
pre { background: var(--code); padding: 0.8rem 1rem; border-radius: 4px; overflow-x: auto; }
pre code { padding: 0; background: none; }
table { border-collapse: collapse; margin: 1rem 0; display: block; overflow-x: auto; }
th, td { text-align: left; padding: 0.3rem 0.9rem 0.3rem 0; border-bottom: 1px solid var(--rule); }
/* Figures keep their printed size. Wide ones break out of the text column
   and scroll horizontally rather than shrink. */
.figure {
  overflow-x: auto; margin: 1.25rem 0;
  width: max-content; max-width: calc(100vw - 32px);
  position: relative; left: 50%; transform: translateX(-50%);
}
.figure img { display: block; max-width: none; height: auto; }
.note { color: var(--muted); font-size: 0.9rem; }
.ruler { width: 1in; border-top: 2px solid var(--ink); margin: 0.4rem 0 0.2rem; }
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MVDLib style guide</title>
<style>{css}</style>
</head>
<body>
{body}
<p class="note">Built from <a href="https://github.com/mvondomaros-lab/mvdlib/blob/main/docs/styleguide.md">docs/styleguide.md</a>.</p>
</body>
</html>
"""

REPO = "https://github.com/mvondomaros-lab/mvdlib/blob/main/docs"

IMG = re.compile(r'(?:<p>)?<img src="([^"]+)" alt="([^"]*)"[^>]*>(?:</p>)?')


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(DOCS / "figures", out / "figures")
    (out / "fonts").mkdir()
    for font in FONTS.glob("*.ttf"):
        shutil.copy(font, out / "fonts")

    body = markdown.markdown((DOCS / "styleguide.md").read_text(), extensions=["fenced_code", "tables"])

    def figure(match: re.Match) -> str:
        src, alt = match[1], match[2]
        with Image.open(DOCS / src) as image:
            width, height = (round(n / DPI * 96) for n in image.size)
        return (
            f'<div class="figure"><img src="{src}" alt="{alt}" '
            f'width="{width}" height="{height}"></div>'
        )

    body = IMG.sub(figure, body)
    # Relative links point into the repository, which the site does not contain.
    body = re.sub(r'href="(?!https?:|#)([^"]+)"', rf'href="{REPO}/\1"', body)
    ruler = (
        '<div class="ruler"></div><p class="note">This bar is 1 in long at 100 % zoom. '
        "If it measures differently on your screen, scale your expectations by the same factor; "
        "all figures share one scale either way.</p>"
    )
    body = body.replace("</h1>", "</h1>\n" + ruler, 1)
    (out / "index.html").write_text(TEMPLATE.format(css=CSS, body=body))


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS.parent / "site")
