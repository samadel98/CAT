"""Lock the database files, so that several processes can share a database.

All functions and classes of Data-CAT that read or write the .csv and .hdf5 files of a
:class:`~dataCAT.Database` hold a lock file in its directory while doing so.
Several CAT runs can thus use the same database at the same time (*e.g.* jobs on a cluster):
they wait for each other instead of overwriting each other's results.

Index
-----
.. currentmodule:: dataCAT.locking
.. autosummary::
    LOCK_NAME
    get_lock
    open_hdf5

API
---
.. autodata:: LOCK_NAME
.. autofunction:: get_lock
.. autofunction:: open_hdf5

"""

from __future__ import annotations

import os
from functools import lru_cache
from contextlib import contextmanager
from typing import Any, Iterator, Union

import h5py
from filelock import FileLock

__all__ = ['LOCK_NAME', 'get_lock', 'open_hdf5']

_PathType = Union[str, bytes, 'os.PathLike[str]', 'os.PathLike[bytes]']

#: The name of the lock file in the directory of a database.
LOCK_NAME = 'database.lock'


@lru_cache(maxsize=None)
def _get_lock(dirname: str) -> FileLock:
    return FileLock(os.path.join(dirname, LOCK_NAME))


def get_lock(dirname: _PathType) -> FileLock:
    """Return the lock of the database in the directory **dirname**.

    The lock can be acquired several times by the same thread (*e.g.* to open the .hdf5 file
    while the .csv file is open); other threads and processes wait until it is fully released.

    Examples
    --------
    .. code:: python

        >>> from dataCAT.locking import get_lock

        >>> path = str(...)  # doctest: +SKIP
        >>> with get_lock(path):  # doctest: +SKIP
        ...     ...  # Read or modify the database in `path`

    """
    return _get_lock(os.path.abspath(os.fsdecode(dirname)))


@contextmanager
def open_hdf5(
    filename: _PathType, mode: str = 'r', **kwargs: Any
) -> Iterator[h5py.File]:
    r"""Open **filename** with :class:`h5py.File` while holding the lock of its database.

    Parameters
    ----------
    filename : :class:`str`, :class:`bytes` or :class:`os.PathLike`
        The path of the .hdf5 file.
    mode : :class:`str`
        The mode for opening the file.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :class:`h5py.File`.

    """
    with get_lock(os.path.dirname(os.path.abspath(os.fsdecode(filename)))):
        with h5py.File(filename, mode, **kwargs) as f:
            yield f
