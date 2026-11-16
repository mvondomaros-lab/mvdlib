# MVDLib

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/)

The lab's Matplotlib plotting style and plotting helpers. Use it so that
figures across lab projects and publications share one consistent look.

## Installation

### Pixi (recommended)

After configuring the lab Pixi channel (see
[mvondomaros-lab.github.io](https://mvondomaros-lab.github.io)), install with:

```console
pixi add mvdlib
```

### PyPI

```console
pip install mvdlib
```

## Usage

```python
import matplotlib.pyplot as plt

from mvdlib.plots import FigSize, contourplot, kdeplot
from mvdlib.style import BASE, H1, H2, H3, INK, LINES, SEQ, DIV

plt.style.use("mvdlib.style.default")

figsize = FigSize()
fig, ax = plt.subplots(figsize=figsize("single"))
```

- `mvdlib.style`: the `default` Matplotlib style sheet and its colour system
  (`INK`, `BASE`, highlight colours `H1`–`H3`, the `LINES` cycle, and the
  `SEQ`/`DIV` colormaps).
- `mvdlib.plots`: plotting helpers: `FigSize` for publication column
  layouts, `contourplot` and `kdeplot`.

See `examples/plots.ipynb` for a tour.
