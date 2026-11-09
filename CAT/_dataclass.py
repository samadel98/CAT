"""A class with generic (magic) methods, taken from `AssertionLib <https://github.com/nlesc-nano/AssertionLib>`_ 3.2.2.

CAT and Nano-CAT used to import :class:`AbstractDataClass` from AssertionLib, which is no longer
maintained; it is copied here, with shortened docstrings.
AssertionLib is licensed under the Apache License 2.0 (see ``LICENSE-Apache-2.0.txt``),
and was developed by B. F. van Beek.

"""  # noqa: E501

from __future__ import annotations

import copy
import textwrap
from abc import ABCMeta
from functools import wraps
from threading import get_ident
from typing import Any, ClassVar, NoReturn, TypeVar, cast
from collections.abc import Callable, Iterable, Iterator, Mapping

__all__ = ["AbstractDataClass"]

_TT = TypeVar("_TT", bound="_MetaADC")
_AT = TypeVar("_AT", bound="AbstractDataClass")
_FT = TypeVar("_FT", bound=Callable[..., Any])


def recursion_safeguard(fallback: _FT) -> Callable[[_FT], _FT]:
    """Decorate a function such that it calls **fallback** in case of recursive calls.

    Implementation based on :func:`reprlib.recursive_repr`.

    """
    def decorating_function(user_function: _FT) -> _FT:
        running: set[tuple[int, int]] = set()

        @wraps(user_function)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = id(self), get_ident()
            if key in running:
                return fallback(self, *args, **kwargs)

            running.add(key)
            try:
                result = user_function(self, *args, **kwargs)
            finally:
                running.discard(key)
            return result
        return cast(_FT, wrapper)
    return decorating_function


class _MetaADC(ABCMeta):
    _HASHABLE: bool = NotImplemented

    def __new__(
        mcls: type[_TT], name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> _TT:
        cls = super().__new__(mcls, name, bases, namespace)  # type: ignore[misc]
        if not cls._HASHABLE:
            cls.__hash__ = mcls._hash_template1
        else:
            cls.__hash__ = recursion_safeguard(cls._hash_fallback)(mcls._hash_template2)
        return cast(_TT, cls)

    @staticmethod
    def _hash_template1(self: AbstractDataClass) -> NoReturn:
        """Unhashable type; raise a :exc:`TypeError`."""
        raise TypeError(f"Unhashable type: {self.__class__.__name__!r}")

    @staticmethod
    def _hash_template2(self: AbstractDataClass) -> int:
        """Return the hash of this instance, constructed from its (non-private) attributes.

        The :func:`id` of unhashable attributes is used instead of their hash.

        """
        if self._hash:
            return self._hash

        ret = hash(type(self))
        for k, v in self._iter_attrs():
            if k in self._PRIVATE_ATTR:
                continue
            try:
                ret ^= hash((k, v))
            except TypeError:
                ret ^= hash((k, id(v)))

        self._hash = ret
        return ret


class AbstractDataClass(metaclass=_MetaADC):
    """A dataclass with a number of generic pre-defined (magic) methods.

    Provides methods for string conversion (:meth:`__repr__`), comparisons (:meth:`__eq__`),
    hashing (:meth:`__hash__`), copying (:meth:`copy`, :meth:`__copy__` and
    :meth:`__deepcopy__`), dictionary interconversion (:meth:`as_dict` and :meth:`from_dict`)
    and inheriting method docstrings and annotations (:meth:`inherit_annotations`).

    Attributes
    ----------
    _PRIVATE_ATTR : :class:`frozenset[str] <frozenset>` or :class:`set[str] <set>`
        A class variable with the names of private instance variables, which are excluded from
        :meth:`as_dict`, printing and comparisons.
        The set is unfrozen (and added as instance variable) when an instance is created.
    _HASHABLE : :class:`bool`
        A class variable denoting whether or not instances are hashable.
    _hash : :class:`int`
        An attribute for caching the :func:`hash` of this instance (if hashable).

    """

    _PRIVATE_ATTR: set[str] | frozenset[str] = frozenset()
    _HASHABLE: ClassVar[bool] = True

    #: Empty slots which can be filled by subclasses.
    __slots__: str | Iterable[str] = ('__dict__',)

    def __init__(self) -> None:
        """Initialize a :class:`AbstractDataClass` instance."""
        cls = type(self)
        self._PRIVATE_ATTR = {'_PRIVATE_ATTR'}.union(cls._PRIVATE_ATTR)
        if cls._HASHABLE:
            self._PRIVATE_ATTR.add('_hash')
            self._hash: int = 0

    def _iter_attrs(self) -> Iterator[tuple[str, Any]]:
        """Return an iterator over the attributes of this instance, as key/value pairs."""
        return iter(vars(self).items())

    def _hash_fallback(self) -> int:
        """Fallback function for :meth:`__hash__` in case of recursive calls."""
        return id(self)

    def _repr_fallback(self) -> str:
        """Fallback function for :meth:`__repr__` in case of recursive calls."""
        return object.__repr__(self).rstrip('>').rsplit(maxsplit=1)[1]

    @recursion_safeguard(fallback=_repr_fallback)  # type: ignore[arg-type]
    def __repr__(self) -> str:
        """Return a string representation of this instance and its (non-private) attributes."""
        try:
            width = max(len(k) for k, _ in self._str_iterator())
        except ValueError:  # Raised if this instance has no instance variables
            return f'{self.__class__.__name__}()'

        ret = ',\n'.join(self._str(k, v, width, 3+width) for k, v in self._str_iterator())
        indent = ' ' * 4
        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent)}\n)'

    def _str_iterator(self) -> Iterable[tuple[str, Any]]:
        """Return an iterable for the :meth:`__repr__` method."""
        return ((k, v) for k, v in sorted(self._iter_attrs()) if k not in self._PRIVATE_ATTR)

    @staticmethod
    def _str(key: str, value: Any, width: int | None = None, indent: int | None = None) -> str:
        """Return a string representation of a single **key**/**value** pair."""
        key_str = f'{key} = ' if width is None else f'{key:{width}} = '
        if indent is not None:
            value_str = textwrap.indent(repr(value), ' ' * indent)[indent:]
        else:
            value_str = repr(value)
        return f'{key_str}{value_str}'

    def _eq_fallback(self, value: Any) -> bool:
        """Fallback function for :meth:`__eq__` in case of recursive calls."""
        return self is value

    @recursion_safeguard(fallback=_eq_fallback)  # type: ignore[arg-type]
    def __eq__(self, value: Any) -> bool:
        """Check if this instance and **value** have the same type and (non-private) attributes."""
        if type(self) is not type(value):
            return False

        try:
            for k, v1 in self._iter_attrs():
                if k in self._PRIVATE_ATTR:
                    continue
                v2 = getattr(value, k)
                assert self._eq(v1, v2)
        except (AttributeError, AssertionError):
            return False
        else:
            return True

    @staticmethod
    def _eq(v1: Any, v2: Any) -> bool:
        """Return if **v1** and **v2** are equivalent."""
        return cast(bool, v1 == v2)

    def copy(self: _AT, deep: bool = False) -> _AT:
        """Return a shallow or deep copy of this instance."""
        cls = type(self)
        ret: _AT = cls.__new__(cls)
        for k, v in self._iter_attrs():
            setattr(ret, k, copy.deepcopy(v) if deep else v)
        return ret

    def __copy__(self: _AT) -> _AT:
        """Return a shallow copy of this instance; see :meth:`AbstractDataClass.copy`."""
        return self.copy(deep=False)

    def __deepcopy__(self: _AT, memo: dict[int, Any] | None = None) -> _AT:
        """Return a deep copy of this instance; see :meth:`AbstractDataClass.copy`."""
        return self.copy(deep=True)

    def as_dict(self, return_private: bool = False) -> dict[str, Any]:
        """Return a dictionary with (shallow copies of) the (non-private) attributes."""
        if return_private:
            return {k: copy.copy(v) for k, v in self._iter_attrs()}
        return {k: copy.copy(v) for k, v in self._iter_attrs() if k not in self._PRIVATE_ATTR}

    @classmethod
    def from_dict(cls: type[_AT], dct: Mapping[str, Any]) -> _AT:
        """Construct an instance of this class from a dictionary with keyword arguments."""
        return cls(**dct)

    @classmethod
    def inherit_annotations(cls) -> Callable[[_FT], _FT]:
        """Return a decorator for inheriting the annotations and docstring of a method.

        The decorated method of a subclass inherits them from the identically named method of
        :class:`AbstractDataClass`; references to the latter are replaced by the subclass.

        """
        def decorator(func: _FT) -> _FT:
            cls_func = getattr(cls, func.__name__)
            sub_cls_name: str = func.__qualname__.split('.')[0]

            if not getattr(func, '__annotations__', None):
                func.__annotations__ = dct = getattr(cls_func, '__annotations__', {}).copy()
                if dct.get('return') in {cls, cls.__name__}:
                    dct['return'] = sub_cls_name

            if func.__doc__ is None and cls_func.__doc__ is not None:
                func.__doc__ = cls_func.__doc__.replace(cls.__name__, sub_cls_name)
            return func
        return decorator
