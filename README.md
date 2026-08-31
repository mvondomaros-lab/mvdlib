# MVDLib

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/)

A loose collection of library functions. Acts as an incubator for future projects. Use at your own risk.

## Installation

### Pixi (recommended)

After configuring the lab Pixi channel (see
[mvondomaros-lab.github.io](https://mvondomaros-lab.github.io)), install the
interactive bundle with:

```console
pixi add mvdlib
```

`mvdlib` is an all-in-one metapackage for interactive work. Libraries should
depend on the narrowest feature package; every feature package installs
`mvdlib-core` automatically.

| Need | Pixi dependency |
| --- | --- |
| Base package only | `mvdlib-core` |
| Styles and plotting helpers | `mvdlib-plots` |
| FFT-based statistics | `mvdlib-stats` |
| Diffusion tools | `mvdlib-diffusion` |
| GROMACS EDR support | `mvdlib-gromacs` |
| Everything | `mvdlib` |

For example, a plotting-only project uses `pixi add mvdlib-plots`, while a
diffusion-only project uses `pixi add mvdlib-diffusion`.

### PyPI

The base Python distribution contains only the lightweight core package.
Install functionality with extras:

```console
pip install "mvdlib[plots]"
pip install "mvdlib[stats]"
pip install "mvdlib[diffusion]"
pip install "mvdlib[gromacs]"
pip install "mvdlib[all]"
```
