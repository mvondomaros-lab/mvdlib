from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True, slots=True)
class FigSize:
    """
    Compute Matplotlib figure sizes for publication column layouts.
    """

    single_width: float = 3.35
    double_width: float = 7.0
    height: float = 2.5

    def __post_init__(self) -> None:
        for name in ("single_width", "double_width", "height"):
            value = _validate_dimension(name, getattr(self, name))
            object.__setattr__(self, name, value)

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
        if height is None:
            height = self.height
        else:
            height = _validate_dimension("height", height)
        if layout in (1, "single"):
            width = self.single_width
        elif layout in (2, "double"):
            width = self.double_width
        else:
            raise ValueError("layout must be 1, 2, 'single', or 'double'")

        return width, height


def _validate_dimension(name: str, value: float) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value
