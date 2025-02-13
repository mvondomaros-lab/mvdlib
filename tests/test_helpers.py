from typing import Union

import numpy as np
import pytest
from numpy.typing import NDArray

import mvdlib.helpers
from mvdlib.helpers import cabs2, nextpow2, unwrap


@pytest.mark.parametrize(
    "x, expected",
    [
        (complex(0.0, 0.0), 0.0),
        (complex(1.0, 0.0), 1.0),
        (complex(0.0, 1.0), 1.0),
        (complex(1.0, 1.0), 2.0),
        (complex(-1.0, -1.0), 2.0),
    ],
)
def test_cabs2_scalars(x: complex, expected: float) -> None:
    """Test cabs2 for scalar complex inputs."""
    assert cabs2(x) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.ones(3, dtype=np.complex64), np.ones(3, dtype=np.float64)),
        (np.ones(3, dtype=np.complex128), np.ones(3, dtype=np.float64)),
        (np.array([3 + 4j, 1 + 1j], dtype=np.complex128), np.array([25.0, 2.0])),
    ],
)
def test_cabs2_arrays(x: NDArray[np.complex128], expected: NDArray[np.float64]) -> None:
    """Test cabs2 for array inputs."""
    result = cabs2(x)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (-1, 1),
        (-2, 2),
        (0.1, 1),
        (-0.1, 1),
        (2**31 - 1, 2**31),  # Edge of int32
        (1e-10, 1),  # Very small number
        (1e10, 2**34),  # Very large number
    ],
)
def test_nextpow2(x: Union[int, float], expected: int) -> None:
    """Test nextpow2 for various scalar inputs."""
    assert nextpow2(x) == expected


@pytest.mark.parametrize(
    "x, box, expected",
    [
        (np.array([1.0, 1.1]), 2.0, np.array([1.0, 1.1])),
        (np.array([1.0, 0.9]), 2.0, np.array([1.0, 0.9])),
        (np.array([0.0, 1.0]), 2.0, np.array([0.0, 1.0])),
        (np.array([0.0, 1.9]), 2.0, np.array([0.0, -0.1])),
        (np.array([0.0, 1.1]), 2.0, np.array([0.0, -0.9])),
        (np.array([1.9, 0.0]), 2.0, np.array([1.9, 2.0])),
        (np.array([1.1, 0.0]), 2.0, np.array([1.1, 2.0])),
    ],
)
def test_unwrap(x: NDArray, box: float, expected: NDArray) -> None:
    """Test the trajectory unwrapping function."""
    unwrap(x, box)
    assert x == pytest.approx(expected)
