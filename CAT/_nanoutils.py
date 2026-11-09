"""Utilities taken from `Nano-Utils <https://github.com/nlesc-nano/Nano-Utils>`_ 2.3.5.

CAT, Nano-CAT and Data-CAT used to import the functions and classes below from Nano-Utils,
which is no longer maintained (and some of whose versions fail on NumPy >= 2).
They are copied here, with shortened docstrings, so that none of the three packages depends
on Nano-Utils.
Nano-Utils is licensed under the Apache License 2.0 (see ``LICENSE-Apache-2.0.txt``),
and its NOTICE reads: "This product includes Nano-Utils, software developed by B. F. van Beek".

"""

from __future__ import annotations

import os
import reprlib
from math import comb
from functools import partial
from itertools import combinations
from threading import RLock
from types import TracebackType
from collections import abc
from typing import Any, Generic, NamedTuple, TypeVar
from collections.abc import Collection, Hashable, Iterable, MutableMapping

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from packaging.version import Version

from ._compat import COPY_IF_NEEDED

__all__ = [
    "as_nd_array", "array_combinations", "group_by_values", "split_dict",
    "PartialPrepend", "SetAttr", "VersionInfo", "PathType",
]

_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")

#: An annotation for path-like objects.
PathType = str | bytes | os.PathLike


class VersionInfo(NamedTuple):
    """A :func:`~collections.namedtuple` representing the version of a package."""

    major: int
    minor: int = 0
    micro: int = 0

    @property
    def patch(self) -> int:
        """An alias for :attr:`VersionInfo.micro`."""
        return self.micro

    @classmethod
    def from_str(cls, version: str) -> VersionInfo:
        """Construct a :class:`VersionInfo` from a :pep:`440` version string (*e.g.* ``"0.8.2"``).

        Versions are truncated to (at most) three integers.

        """
        release = Version(version).release[:3]
        return cls(*release)


def group_by_values(iterable: Iterable[tuple[_VT, _KT]]) -> dict[_KT, list[_VT]]:
    """Take an iterable, yielding 2-tuples, and group all first elements by the second.

    .. code:: python

        >>> from CAT._nanoutils import group_by_values
        >>> group_by_values(enumerate(['a', 'a', 'b', 'c', 'b'], 1))
        {'a': [1, 2], 'b': [3, 5], 'c': [4]}

    """
    ret: dict[_KT, list[_VT]] = {}
    for value, key in iterable:
        ret.setdefault(key, []).append(value)
    return ret


def split_dict(
    dct: MutableMapping[_KT, _VT],
    preserve_order: bool = False,
    *,
    keep_keys: None | Iterable[_KT] = None,
    disgard_keys: None | Iterable[_KT] = None,
) -> dict[_KT, _VT]:
    """Pop the items of **dct** not in **keep_keys** (or in **disgard_keys**) into a new dictionary.

    **dct** is modified inplace. **keep_keys** and **disgard_keys** are mutually exclusive.

    .. code:: python

        >>> from CAT._nanoutils import split_dict
        >>> dict1 = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
        >>> split_dict(dict1, keep_keys={1, 2})
        {3: 'c', 4: 'd'}
        >>> dict1
        {1: 'a', 2: 'b'}

    """
    keys: Collection[_KT]
    if keep_keys is None and disgard_keys is None:
        raise TypeError("'keep_keys' and 'disgard_keys' cannot both be unspecified")
    elif keep_keys is None:
        assert disgard_keys is not None
        disgard = set(disgard_keys)
        keys = [k for k in dct if k in disgard] if preserve_order else dct.keys() & disgard
    elif disgard_keys is None:
        keep = set(keep_keys)
        keys = [k for k in dct if k not in keep] if preserve_order else dct.keys() - keep
    else:
        raise TypeError("'keep_keys' and 'disgard_keys' cannot both be specified")
    return {k: dct.pop(k) for k in keys}


class PartialPrepend(partial):
    """A :func:`~functools.partial` subclass where the ``*args`` are appended rather than prepended.

    .. code:: python

        >>> from CAT._nanoutils import PartialPrepend
        >>> func = PartialPrepend(isinstance, float)  # isinstance(..., float)
        >>> func(1.0)
        True

    """

    def __call__(self, /, *args: Any, **keywords: Any) -> Any:
        """Call and return :attr:`~PartialPrepend.func`."""
        keywords = {**self.keywords, **keywords}
        return self.func(*args, *self.args, **keywords)


class SetAttr(Generic[_T1, _T2]):
    """A reusable and reentrant context manager for temporarily changing an attribute's value.

    The value to restore is the one at the time the instance is created.

    .. code:: python

        >>> from types import SimpleNamespace
        >>> from CAT._nanoutils import SetAttr
        >>> obj = SimpleNamespace(a=False)
        >>> with SetAttr(obj, 'a', True):
        ...     print(obj.a)
        True
        >>> obj.a
        False

    """

    __slots__ = ('__weakref__', '_obj', '_name', '_value', '_value_old', '_lock')

    @property
    def obj(self) -> _T1:
        """The to-be modified object."""
        return self._obj

    @property
    def name(self) -> str:
        """The name of the to-be modified attribute."""
        return self._name

    @property
    def value(self) -> _T2:
        """The value assigned to the :attr:`name` attribute of :attr:`obj`."""
        return self._value

    def __init__(self, obj: _T1, name: str, value: _T2) -> None:
        """Initialize the context manager."""
        self._obj = obj
        self._name = name
        self._value = value
        self._value_old: _T2 = getattr(obj, name)
        self._lock = RLock()

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        """Implement :func:`repr(self)<repr>`."""
        obj = object.__repr__(self.obj)
        value = reprlib.repr(self.value)
        return f'{self.__class__.__name__}(obj={obj}, name={self.name!r}, value={value})'

    def __enter__(self) -> None:
        """Enter the context manager, modify :attr:`obj`."""
        with self._lock:
            setattr(self.obj, self.name, self.value)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager, restore :attr:`obj`."""
        with self._lock:
            setattr(self.obj, self.name, self._value_old)


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
