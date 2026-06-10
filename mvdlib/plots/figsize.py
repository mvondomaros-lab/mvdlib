from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FigSize:
    """
    Compute Matplotlib figure sizes for publication column layouts.
    """

    single_width: float = 3.35
    double_width: float = 7.0
    height: float = 2.5

    def __call__(
        self,
        layout: int | str = 1,
        height: float | None = None,
    ) -> tuple[float, float]:
        """
        Compute the figure size.

        Parameters
        ----------
        layout
            Figure width. Accepted values are ``1`` or ``"single"`` for a
            single-column figure and ``2`` or ``"double"`` for a double-column
            figure.
        height
            Figure height in inches. Defaults to the instance height.

        Returns
        -------
        tuple of float
            Figure width and height in inches.
        """
        height = self.height if height is None else height
        if height <= 0.0:
            raise ValueError("height must be positive")
        if layout in (1, "single"):
            width = self.single_width
        elif layout in (2, "double"):
            width = self.double_width
        else:
            raise ValueError("layout must be 1, 2, 'single', or 'double'")

        return width, height
