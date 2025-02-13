from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


class FigSize:
    """
    Callable object for returning figure dimensions.
    """

    def __init__(
        self, w: float = 1.5, h: float = 1.63, w0: float = 1.0, h0: float = 0.47
    ):
        """
        Initialize a FigSize object.

        :param w: Width of one panel.
        :param h: Height of one panel.
        :param w0: Width offset.
        :param h0: Height offset.
        """
        self.w = w
        self.h = h
        self.w0 = w0
        self.h0 = h0

    def __call__(self, nrows: int | float, ncols: int | float) -> NDArray:
        """
        Calculate the figure size.

        Uses the following formulas:
            width = w0 + ncols * w
            height = h0 + nrows * h

        :param nrows: The number of rows.
        :param ncols: The number of columns.

        :return: The figure size.
        """
        width = self.w0 + self.w * ncols
        height = self.h0 + self.h * nrows
        return np.array([width, height])
