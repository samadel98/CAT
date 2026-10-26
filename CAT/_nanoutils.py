"""Utilities taken from `Nano-Utils <https://github.com/nlesc-nano/Nano-Utils>`_ 2.3.5.

CAT used to import the functions below from Nano-Utils, whose versions fail on NumPy >= 2.
They are copied here, with shortened docstrings, so that they work on both NumPy 1.x and 2.x.
Nano-Utils is licensed under the Apache License 2.0 (see ``LICENSE-Apache-2.0.txt``),
and its NOTICE reads: "This product includes Nano-Utils, software developed by B. F. van Beek".

"""

from math import comb
from itertools import combinations
from collections import abc
from typing import Any
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ._compat import COPY_IF_NEEDED

__all__ = ["as_nd_array", "array_combinations"]


def as_nd_array(
    array: Iterable[Any] | ArrayLike,
    dtype: DTypeLike,
    ndmin: int = 1,
    copy: bool = False,
) -> np.ndarray:
    """Construct a numpy array from an iterable or array-like object.

    Unlike Nano-Utils, ``copy=False`` copies when needed rather than raising on NumPy >= 2.

    """
    try:
        return np.array(array, dtype=dtype, ndmin=ndmin, copy=True if copy else COPY_IF_NEEDED)
    except TypeError:
        if not isinstance(array, abc.Iterable):
            raise
        ret: np.ndarray = np.fromiter(array, dtype=dtype)
        return ret.reshape(ret.shape + (ndmin - ret.ndim) * (1,))


def array_combinations(array: ArrayLike, r: int = 2, axis: int = -1) -> np.ndarray:
    r"""Construct an array with all :func:`~itertools.combinations` of **ar** along a given axis.

    Parameters
    ----------
    array : array-like, shape :math:`(m, \dotsc)`
        An :math:`n` dimensional array-like object.
    r : :class:`int`
        The length of each combination.
    axis : :class:`int`
        The axis used for constructing the combinations.

    Returns
    -------
    :class:`numpy.ndarray`, shape :math:`(k, \dotsc, r)`
        A :math:`n+1` dimensional array with all **ar** combinations (of length ``r``)
        along the user-specified **axis**.
        The variable :math:`k` herein represents the number of combinations:
        :math:`k = \dfrac{m! / r!}{(m-r)!}`.

    """  # noqa: E501
    ar = np.array(array, ndmin=1, copy=COPY_IF_NEEDED)
    n = ar.shape[axis]

    # Identify the number of combinations
    if not (0 <= r <= n):
        raise ValueError(f"'r' ({r!r}) expects a positive integer larger than or equal to the "
                         f"length of 'array' axis {axis!r} ({n!r})")
    combinations_len = comb(n, r)

    # Define the shape of the to-be returned array
    _shape = list(ar.shape)
    del _shape[axis]
    shape = (combinations_len,) + tuple(_shape) + (r,)

    # Create, fill and return the new array
    ret = np.empty(shape, dtype=ar.dtype)
    for i, idx in enumerate(combinations(range(n), r=r)):
        ret[i] = ar.take(idx, axis=axis)
    return ret
