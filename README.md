# MVDLib

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/)

A loose collection of library functions. Acts as an incubator for future projects. Use at your own risk.

## Installation

1.  Setup the the [uv](https://docs.astral.sh/uv/) package manager.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    Add the following lines to your `uv.toml` (usually located under `~/.config/uv/uv.toml`).
    ```toml
    [[index]]
    name = "mvondomaros-lab"
    url = "https://mvondomaros-lab.github.io"
    ```
2.  Add `mvdlib` to your project.
    ```bash
    uv add mvdlib
    ```
    
    Or use it with uv tools.
    
    ```bash
    uv tool install marimo --with "numba,numpy,scipy,pandas,matplotlib,seaborn,MDAnalysis,mvdlib"
    ```
    
