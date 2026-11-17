# Changelog

## 0.22.0

MVDLib is now a plotting-only library: the lab's Matplotlib style and plotting
helpers. This release breaks compatibility.

### Removed

- The `diffusion`, `timeseries`, and `gromacs` modules, and the internal
  `_core` and `types` modules. They remain available in version 0.21.0.
- The `mvdlib-core`, `mvdlib-plots`, `mvdlib-stats`, `mvdlib-diffusion`, and
  `mvdlib-gromacs` conda packages and the pip extras. Depend on `mvdlib`.
- The seaborn dependency.
- `examples/plots.ipynb`, replaced by the style guide (`docs/styleguide.md`).

### Changed

- Retuned the line colours for colour vision deficiencies and greyscale:
  `BASE` `#236DA6`, `H1` `#8E274B`, `H2` `#079281`, `H3` `#BC8C06`.
- The colormaps are now `mvdlib_seq` and `mvdlib_div` (formerly seaborn's
  `crest` and `vlag`, which they reproduce).
- Math text is typeset in Fira Sans, matching the rest of the text.
- Data lines are thicker (1.0 pt) than axes and ticks (0.6 pt); the grid is
  solid and thin.
- Minor ticks are off by default.
- `pcolor.shading` is `auto` and `hist.bins` is `auto`.
- `FigSize` is replaced by `subplots`, which fixes the panel size (default
  2.0 × 1.5 in) instead of the figure size, so panels are equally large in
  single- and multi-panel figures.
- Figure-level labels and titles (`fig.supxlabel`, `fig.suptitle`, ...) use
  the axis label size.
- `contourplot` is now `plot_contour`, with `clabels` and `grid` options.
- `kdeplot` is now `plot_kde`. Its default grid extends three bandwidths
  beyond the data, and it can fill the area under the curve.

### Added

- Fira Sans is bundled and registered on import.
- PDF and PostScript output embed TrueType fonts, SVG output keeps text as
  text, and raster output uses 600 dpi.
- `GRID` colour constant.
- `axis_label`, `panel_labels`, and `set_nticks` helpers.
- `colorbar` helper and the `ACS_SINGLE_COLUMN` and `ACS_DOUBLE_COLUMN`
  widths.
- The style guide website at https://mvondomaros-lab.github.io/mvdlib/, which
  shows every figure at its printed size.
