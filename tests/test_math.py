import numpy as np
import pytest
from numpy.typing import NDArray

# noinspection PyProtectedMember
from mvdlib._core.math import cabs2, nextpow2, prevpow2


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
    result = cabs2(x)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (5, 4),
        (8, 8),
        (-1, 1),
        (-2, 2),
        (0.1, 1),
        (-0.1, 1),
        (2**31 - 1, 2**30),
        (1e-10, 1),
        (1e10, 2**33),
    ],
)
def test_prevpow2(x: int | float, expected: int) -> None:
    assert prevpow2(x) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (1.1, 2),
        (2.1, 4),
        (-2.1, 4),
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
def test_nextpow2(x: int | float, expected: int) -> None:
    assert nextpow2(x) == expected
