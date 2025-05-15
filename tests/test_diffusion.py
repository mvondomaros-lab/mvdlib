from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from mvdlib import diffusion


@pytest.mark.parametrize(
    "x, nc, expected",
    [
        (np.zeros(5), None, np.zeros(5)),
        (np.ones(5), None, np.full(5, 2.0)),
        (np.arange(5), None, np.array([12.0, 11.0, 34.0 / 3.0, 13.0, 16.0])),
        (np.zeros(5), 2, np.zeros(2)),
        (np.ones(5), 2, np.full(2, 2.0)),
        (np.arange(5), 2, np.array([12.0, 11.0])),
    ],
)
def test_mssq(x: NDArray, nc: Optional[int], expected: NDArray) -> None:
    """Test the mean sum of squares function."""
    nc = nc or x.size
    assert diffusion._mssq(x, nc) == pytest.approx(expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.zeros(5), np.zeros(5)),
        (np.ones(5), np.zeros(5)),
        (np.arange(5), np.arange(5) ** 2),
    ],
)
def test_msd(x: NDArray, expected: NDArray) -> None:
    """Test the mean squared displacement function."""
    msd = diffusion.msd(x)
    assert msd == pytest.approx(expected)
