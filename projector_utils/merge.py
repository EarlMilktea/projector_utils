"""Utilities to group and ungroup tensor axes."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


def group(arr: npt.ArrayLike, begin: SupportsIndex, end: SupportsIndex) -> npt.NDArray[Any]:
    """Merge axes in the half-open range ``[begin, end)`` into one.

    Parameters
    ----------
    arr
        Input array.
    begin
        First axis index to merge (inclusive).
    end
        Axis index after the last axis to merge (exclusive).

    Returns
    -------
    numpy.ndarray
        Array with axes ``begin`` through ``end - 1`` flattened into a single
        axis.

    Raises
    ------
    ValueError
        If ``begin`` or ``end`` are out of range, or ``begin >= end``.

    Notes
    -----
    The merged axis preserves C-order (row-major) element ordering of the
    merged axes.
    """
    arr = np.asarray(arr)
    begin = operator.index(begin)
    end = operator.index(end)
    if begin < 0:
        begin += arr.ndim
    if end < 0:
        end += arr.ndim
    if not (0 <= begin < end <= arr.ndim):
        msg = "begin and end must satisfy 0 <= begin < end <= arr.ndim after normalization."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:begin], -1, *arr.shape[end:])


def ungroup(arr: npt.ArrayLike, target: SupportsIndex, split: Sequence[SupportsIndex]) -> npt.NDArray[Any]:
    """Split the specified axis into the given shape.

    Parameters
    ----------
    arr
        Input array.
    target
        Axis index to split.
    split
        Target shape for the axis being split.

    Returns
    -------
    numpy.ndarray
        Array with the target axis reshaped to ``split``.

    Raises
    ------
    ValueError
        If ``target`` is out of range, or the target axis size does not match
        ``math.prod(split)``.

    Notes
    -----
    The split uses C-order (row-major) when expanding the target axis into
    ``split``.
    """
    arr = np.asarray(arr)
    target = operator.index(target)
    if target < 0:
        target += arr.ndim
    if not (0 <= target < arr.ndim):
        msg = "target must be between 0 and arr.ndim - 1 after normalization."
        raise ValueError(msg)
    split = tuple(operator.index(i) for i in split)
    nl = int(arr.shape[target])
    if nl != math.prod(split):  # type: ignore[arg-type]
        msg = f"Cannot ungroup: {nl} -> {split}."
        raise ValueError(msg)
    return arr.reshape(*arr.shape[:target], *split, *arr.shape[target + 1 :])
