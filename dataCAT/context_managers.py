"""A module which holds context managers for the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.context_managers
.. autosummary::
    OpenLig
    OpenQD

API
---
.. autoclass:: OpenLig
.. autoclass:: OpenQD

"""

from os.path import abspath, dirname
from abc import abstractmethod, ABCMeta
from types import TracebackType
from typing import (
    Optional, Any, Generic, TypeVar, Tuple, Type, AnyStr, Dict, TYPE_CHECKING, Union, final
)

import numpy as np
import pandas as pd
from filelock import FileLock

from .df_proxy import DFProxy
from .locking import get_lock

if TYPE_CHECKING:
    from os import PathLike  # noqa: F401

__all__ = ['OpenLig', 'OpenQD']

T = TypeVar('T')
ST = TypeVar('ST', bound='FileManagerABC')


class FileManagerABC(Generic[AnyStr, T], metaclass=ABCMeta):
    """An abstract baseclass for opening and closing the various database components."""

    __slots__ = ('__weakref__', '_filename', '_write', '_db', '_hash')

    _db: T

    @property
    def filename(self) -> AnyStr:
        """:data:`~typing.AnyStr`: Get the name of the to-be opened file."""
        return self._filename

    @property
    def write(self) -> bool:
        """:class:`bool`: Get whether or not :attr:`.filename` should be written to when closing the context manager."""  # noqa: E501
        return self._write

    @property
    def lock(self) -> FileLock:
        """:class:`filelock.FileLock`: Get the lock of the database, which is held while the context manager is open."""  # noqa: E501
        return get_lock(dirname(self.filename))

    @final
    def __init__(self, filename: Union[AnyStr, 'PathLike[AnyStr]'], write: bool = True) -> None:
        """Initialize the file context manager.

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes` or :class:`os.PathLike`
            A path-like object.
        write : :class:`bool`
            Whether or not the database file should be updated after closing this instance.


        :rtype: :data:`None`

        """
        self._filename: AnyStr = abspath(filename)
        self._write: bool = write

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False
        return self.filename == value.filename and self.write is value.write

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        return f'{self.__class__.__name__}(filename={self.filename!r}, write={self.write!r})'

    def __reduce__(self: ST) -> Tuple[Type[ST], Tuple[AnyStr, bool]]:
        """A helper function for :mod:`pickle`."""
        cls = type(self)
        return cls, (self.filename, self.write)

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self

    def __deepcopy__(self: ST, memo: Optional[Dict[int, Any]] = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            self._hash: int = hash(self.__reduce__())
            return self._hash

    @abstractmethod
    def __enter__(self) -> T:
        """Enter the context manager; open, store and return the database."""
        raise NotImplementedError("Trying to call an abstract method")
        self._db: T = ...
        return self._db

    @abstractmethod
    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Exit the context manager; close and, optionally, write the database."""
        raise NotImplementedError("Trying to call an abstract method")
        del self._db


class OpenLig(FileManagerABC[AnyStr, DFProxy]):
    """Context manager for opening and closing the ligand database (:attr:`Database.csv_lig`)."""

    def __enter__(self) -> DFProxy:
        """Open the :class:`.OpenLig` context manager, importing :attr:`.df`."""
        # Open the .csv file; other processes have to wait until it is closed again
        self.lock.acquire()
        try:
            dtype = {'hdf5 index': np.int64, 'formula': str, 'settings': str, 'opt': bool}
            self._db = df = DFProxy(
                pd.read_csv(self.filename, index_col=[0, 1], header=[0, 1], dtype=dtype)
            )

            # Fix the columns
            idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
            df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        except BaseException:
            self.lock.release()
            raise
        return df

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Close the :class:`.OpenLig` context manager, exporting :attr:`.df`."""
        try:
            if self.write:
                self._db.to_csv(self.filename)
        finally:
            del self._db
            self.lock.release()


class OpenQD(FileManagerABC[AnyStr, DFProxy]):
    """Context manager for opening and closing the QD database (:attr:`.Database.csv_qd`)."""

    def __enter__(self) -> DFProxy:
        """Open the :class:`.OpenQD` context manager, importing :attr:`.df`."""
        # Open the .csv file; other processes have to wait until it is closed again
        self.lock.acquire()
        try:
            dtype = {'hdf5 index': np.int64, 'settings': str, 'opt': bool}
            self._db = df = DFProxy(
                pd.read_csv(self.filename, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype)
            )

            # Fix the columns
            idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
            df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        except BaseException:
            self.lock.release()
            raise
        return df

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Close the :class:`.OpenQD` context manager, exporting :attr:`.df`."""
        try:
            if self.write:
                self._db.to_csv(self.filename)
        finally:
            del self._db
            self.lock.release()
