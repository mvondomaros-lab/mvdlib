from __future__ import annotations

import typing

import numpy as np
import pyedr

if typing.TYPE_CHECKING:
    from typing import List, Dict
    from numpy.typing import NDArray


def concat_chunks(*arrays: NDArray, contiguous: bool = False) -> NDArray:
    """
    Concatenate one or more 1D NumPy arrays.

    :param arrays: One or more 1D NumPy arrays to concatenate.
    :param contiguous: If True, concatenation uses the full first array
                       and slices each subsequent array from index 1 onward.
                       If False, all arrays are concatenated fully.
    :return: A 1D NumPy array containing the concatenated data.
    """

    if not arrays:
        return np.array([])

    parts = [arrays[0]]

    for arr in arrays[1:]:
        parts.append(arr[1:] if contiguous else arr)

    return np.concatenate(parts)


def read_edrs(
    *paths, fields: List[str], contiguous: bool = True
) -> Dict[str, NDArray[np.float64]]:
    if not paths or not fields:
        return {}

    for path in paths:
        edr = pyedr.edr_to_dict(path)
