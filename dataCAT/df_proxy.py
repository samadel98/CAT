"""A module which holds the :class:`DFProxy` class, a mutable collection for holding Pandas DataFrames.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    DFProxy

API
---
.. autoclass:: DFProxy

"""  # noqa: E501

from itertools import chain
from typing import NoReturn, Tuple, Dict, Any, FrozenSet, TypeVar, Type, ClassVar, cast, final
from textwrap import indent

import pandas as pd
from pandas.core.generic import NDFrame

__all__ = ['DFProxy']

TT = TypeVar('TT', bound='_DFMeta')


class _DFMeta(type):
    MAGIC: FrozenSet[str] = frozenset({
        '__array__', '__abs__', '__add__', '__and__', '__contains__', '__eq__',
        '__floordiv__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__',
        '__ifloordiv__', '__imod__', '__imul__', '__ior__', '__ipow__', '__isub__', '__iter__',
        '__itruediv__', '__ixor__', '__le__', '__len__', '__lt__', '__matmul__', '__mod__',
        '__mul__', '__or__', '__pow__', '__setitem__', '__sub__', '__truediv__', '__xor__',
        '__contains__', '__iter__', '__len__'
    })

    SETTERS: FrozenSet[str] = frozenset({
        'index', 'columns'
    })

    # Should be implemented by the actual (non-meta) classes
    NDTYPE: Type[NDFrame] = NotImplemented

    @staticmethod
    def _construct_getter(cls_name: str, func_name: str, module: str = __name__) -> property:
        """Helper function for :meth:`_DFMeta.__new__`; constructs a :class:`property` getter."""
        def getter(self):
            return getattr(self.ndframe, func_name)

        getter.__doc__ = f"""Get :meth:`pandas.DataFrame.{func_name}` of :attr:`{cls_name}.ndframe`."""  # noqa: E501
        getter.__name__ = func_name
        getter.__qualname__ = f'{cls_name}.{func_name}'
        getter.__module__ = module
        return property(getter)

    @staticmethod
    def _construct_setter(prop: property, cls_name: str, func_name: str,
                          module: str = __name__) -> property:
        """Helper function for :meth:`_DFMeta.__new__`; constructs a :class:`property` setter."""
        def setter(self, value):
            return setattr(self.ndframe, func_name, value)

        setter.__doc__ = f"""Set :meth:`pandas.DataFrame.{func_name}` of :attr:`{cls_name}.ndframe`."""  # noqa: E501
        setter.__name__ = func_name
        setter.__qualname__ = f'{cls_name}.{func_name}'
        setter.__module__ = module
        return prop.setter(setter)

    def __new__(mcls: Type[TT], name: str, bases: Tuple[type, ...],  # noqa: N804
                namespace: Dict[str, Any]) -> TT:
        """Construct a new :class:`_DFMeta` instance."""
        cls = cast(TT, super().__new__(mcls, name, bases, namespace))
        module = cls.__module__

        # Validation
        try:
            ndtype = cls.NDTYPE
            assert isinstance(ndtype, type) and issubclass(ndtype, NDFrame)
        except (AttributeError, AssertionError) as ex:
            raise TypeError(f"{name}.NDTYPE expectes an NDFrame subclass") from ex

        # Construct properties linking to attributes of **cls**
        name_iterator = (i for i in dir(ndtype) if not i.startswith('_'))
        for func_name in chain(mcls.MAGIC, name_iterator):
            getter = mcls._construct_getter(name, func_name, module=module)
            setattr(cls, func_name, getter)

        # Construct setters
        for func_name in mcls.SETTERS:
            getter = getattr(cls, func_name)
            setter = mcls._construct_setter(getter, name, func_name, module=module)
            setattr(cls, func_name, setter)
        return cls


@final
class DFProxy(metaclass=_DFMeta):
    """A mutable wrapper providing a view of the underlying dataframes.

    Attributes
    ----------
    ndframe : :class:`pandas.DataFrame`
        The embedded DataFrame.

    """

    __slots__ = ('__weakref__', 'ndframe')

    #: The type of :class:`~pandas.core.generic.NDFrame` subclass contained within this class.
    NDTYPE: ClassVar[Type[pd.DataFrame]] = pd.DataFrame

    #: The embedded DataFrame.
    ndframe: pd.DataFrame

    def __init__(self, ndframe: pd.DataFrame) -> None:
        """Initialize a new instance.

        Parameters
        ----------
        ndframe : :class:`pandas.DataFrame`
            A Pandas DataFrame (see :attr:`DFProxy.df`).


        :rtype: :data:`None`

        """
        self.ndframe = ndframe

    def __repr__(self) -> str:
        """Implement :func:`repr(self)<repr>`."""
        return f'{self.__class__.__name__}(ndframe={object.__repr__(self.ndframe)})'

    def __str__(self) -> str:
        """Implement :class:`str(self)<str>`."""
        df = indent(str(self.ndframe), 4 * ' ', predicate=bool)
        return f'{self.__class__.__name__}(\n{df}\n)'

    def __reduce__(self) -> NoReturn:
        """Helper function for :mod:`pickle`.

        Warnings
        --------
        This operation will raise a :exc:`TypeError`.

        """
        raise TypeError(f"can't pickle {self.__class__.__name__} objects.")
