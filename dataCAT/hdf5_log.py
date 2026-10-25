"""A module related to logging and hdf5.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    create_hdf5_log
    update_hdf5_log
    reset_hdf5_log
    log_to_dataframe

API
---
.. autofunction:: create_hdf5_log
.. autofunction:: update_hdf5_log
.. autofunction:: reset_hdf5_log
.. autofunction:: log_to_dataframe

"""

from __future__ import annotations

from typing import Sequence, Tuple, Optional, Any, TYPE_CHECKING
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from . import CAT_VERSION, NANOCAT_VERSION, DATACAT_VERSION
from .dtype import DT_DTYPE, VERSION_DTYPE, MSG_DTYPE, INDEX_DTYPE

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'create_hdf5_log', 'update_hdf5_log', 'reset_hdf5_log', 'log_to_dataframe'
]

_VERSION = np.array([CAT_VERSION, NANOCAT_VERSION, DATACAT_VERSION], dtype=VERSION_DTYPE)  # type: ignore  # noqa: E501
_VERSION.setflags(write=False)

_VERSION_NAMES = np.array(['CAT', 'Nano-CAT', 'Data-CAT'], dtype=np.bytes_)
_VERSION_NAMES.setflags(write=False)


LOG_DOC = """A h5py Group for logging database modifications.

Attributes
----------
date : dataset
    A dataset for denoting dates and times when the database was modified.
    Used as dimensional scale for :code:`group['index'].dims[0]` and
    :code:`group['version'].dims[0]`.
version : dataset
    A dataset keeping track of (user-specified) package versions.
version_names : dataset
    A dataset with the names of the packages whose versions are displayed in **version**.
    Used as dimensional scale for :code:`group['version'].dims[1]`.
message : dataset
    A dataset holding user-specified modification messages.
index : dataset
    A dataset with the indices of which elements in the database were modified.

n : attribute
    An attribute with the index of the next to-be set dataset element.
n_step : attribute
    An attribute with the increment in which the length of each dataset should be
    increased in the case of :code:`n >= len(dataset)`.
    Only relevant when :code:`clear_when_full = False`.
clear_when_full : :class:`bool`
    Whether or not to delete and recreate the dataset when it's full.
    Otherwise its length be increased by **n_step**.
date_created : attribute
    An attribute with the date and time from when this logger was created.
version_created : attribute
    An attribute with the versions of a set of user-specified packages from when
    this logger was created.

"""


def _get_now() -> np.recarray:
    now = datetime.now()
    tup = tuple(getattr(now, k) for k in DT_DTYPE.fields.keys())  # type: ignore[union-attr]
    return np.rec.array(tup, dtype=DT_DTYPE)


def create_hdf5_log(file: h5py.Group,
                    n_entries: int = 100,
                    clear_when_full: bool = False,
                    version_names: Sequence[str] | Sequence[bytes] | np.ndarray = _VERSION_NAMES,
                    version_values: Sequence[Tuple[int, int, int]] | np.ndarray = _VERSION,
                    **kwargs: Any) -> h5py.Group:
    r"""Create a hdf5 group for logging database modifications.

    The logger Group consists of four main datasets:

    * ``"date"``: Denotes dates and times for when the database is modified.
    * ``"version"``: Denotes user-specified package versions for when the database is modified.
    * ``"version_names"`` : See the **version_names** parameter.
    * ``"message"``: Holds user-specified modification messages.
    * ``"index"``: Denotes indices of which elements in the database were modified.

    Examples
    --------
    .. testsetup:: python

        >>> import os
        >>> from dataCAT.testing_utils import HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    .. code:: python

        >>> import h5py
        >>> from dataCAT import create_hdf5_log

        >>> hdf5_file = str(...)  # doctest: +SKIP
        >>> with h5py.File(hdf5_file, 'a') as f:
        ...     group = create_hdf5_log(f)
        ...
        ...     print('group', '=', group)
        ...     for name, dset in group.items():
        ...         print(f'group[{name!r}]', '=', dset)
        group = <HDF5 group "/logger" (5 members)>
        group['date'] = <HDF5 dataset "date": shape (100,), type "|V11">
        group['version'] = <HDF5 dataset "version": shape (100, 3), type "|V3">
        group['version_names'] = <HDF5 dataset "version_names": shape (3,), type "|S8">
        group['message'] = <HDF5 dataset "message": shape (100,), type "|O">
        group['index'] = <HDF5 dataset "index": shape (100,), type "|O">

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    file : :class:`h5py.File` or :class:`h5py.Group`
        The File or Group where the logger should be created.
    n_entries : :class:`int`
        The initial number of entries in each to-be created dataset.
        In addition, everytime the datasets run out of available slots their length
        will be increased by this number (assuming :data:`clear_when_full = False<False>`).
    clear_when_full : :class:`bool`
        If :data:`True`, delete the logger and create a new one whenever it is full.
        Increase the size of each dataset by **n_entries** otherwise.
    version_names : :class:`Sequence[str or bytes]<typing.Sequence>`
        A sequence consisting of strings and/or bytes representing the
        names of the to-be stored package versions.
        Should be of the same length as **version_values**.
    version_values : :class:`Sequence[Tuple[int, int, int]]<typing.Sequence>`
        A sequence with 3-tuples, each tuple representing a package version associated with
        its respective counterpart in **version_names**.
    \**kwargs : :data:`~Any`
        Further keyword arguments for the h5py :meth:`~h5py.Group.create_dataset` function.

    Returns
    -------
    :class:`h5py.Group`
        The newly created ``"logger"`` group.

    """
    m = len(version_values)

    if n_entries < 1:
        raise ValueError(f"'n_entries' must ba larger than 1; observed value: {n_entries!r}")
    elif m < 1:
        raise ValueError(f"'version_values' must be larger than 1; observed value: {version_values!r}")  # noqa: E501

    # Set attributes
    grp = file.create_group('logger', track_order=True)
    grp.attrs['__doc__'] = np.bytes_(LOG_DOC)
    grp.attrs['n'] = 0
    grp.attrs['n_step'] = n_entries
    grp.attrs['clear_when_full'] = clear_when_full
    grp.attrs['date_created'] = _get_now()
    grp.attrs['version_created'] = np.asarray(version_values, dtype=VERSION_DTYPE)

    # Set the datasets
    shape1 = (n_entries,)
    shape2 = (n_entries, m)
    data = np.asarray(version_names, dtype=np.bytes_)

    scale1 = grp.create_dataset('date', shape=shape1, maxshape=(None,), dtype=DT_DTYPE, chunks=shape1, **kwargs)  # noqa: E501
    grp.create_dataset('version', shape=shape2, maxshape=(None, m), dtype=VERSION_DTYPE, chunks=shape2, **kwargs)  # noqa: E501
    scale2 = grp.create_dataset('version_names', data=data, shape=(m,), dtype=data.dtype, **kwargs)
    grp.create_dataset('message', shape=shape1, maxshape=(None,), dtype=MSG_DTYPE, chunks=shape1, **kwargs)  # noqa: E501
    grp.create_dataset('index', shape=shape1, maxshape=(None,), dtype=INDEX_DTYPE, chunks=shape1, **kwargs)  # noqa: E501

    # Set dataset scales
    scale1.make_scale('date')
    grp['version'].dims[0].label = 'date'
    grp['version'].dims[0].attach_scale(scale1)
    grp['index'].dims[0].label = 'date'
    grp['index'].dims[0].attach_scale(scale1)
    grp['message'].dims[0].label = 'date'
    grp['message'].dims[0].attach_scale(scale1)

    scale2.make_scale('version_names')
    grp['version'].dims[1].label = 'version_names'
    grp['version'].dims[1].attach_scale(scale2)
    return grp


def update_hdf5_log(
    group: h5py.Group,
    index: ArrayLike,
    message: Optional[str] = None,
    version_values: Sequence[Tuple[int, int, int]] | np.ndarray = _VERSION,
) -> None:
    r"""Add a new entry to the hdf5 logger in **file**.

    Examples
    --------
    .. testsetup:: python

        >>> import os
        >>> from shutil import copyfile
        >>> from dataCAT.testing_utils import HDF5_READ, HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)
        >>> _ = copyfile(HDF5_READ, hdf5_file)

    .. code:: python

        >>> from datetime import datetime

        >>> import h5py
        >>> from dataCAT import update_hdf5_log

        >>> hdf5_file = str(...)  # doctest: +SKIP

        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     group = f['ligand/logger']
        ...
        ...     n = group.attrs['n']
        ...     date_before = group['date'][n]
        ...     index_before = group['index'][n]
        ...
        ...     update_hdf5_log(group, index=[0, 1, 2, 3], message='append')
        ...     date_after = group['date'][n]
        ...     index_after = group['index'][n]

        >>> print(index_before, index_after, sep='\n')
        []
        [0 1 2 3]

        >>> print(date_before, date_after, sep='\n')  # doctest: +SKIP
        (0, 0, 0, 0, 0, 0, 0)
        (2020, 6, 24, 16, 33, 7, 959888)

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    group : :class:`h5py.Group`
        The ``logger`` Group.
    idx : :class:`numpy.ndarray`
        A numpy array with the indices of (to-be logged) updated elements.
    version_values : :class:`Sequence[Tuple[int, int, int]]<typing.Sequence>`
        A sequence with 3-tuples representing to-be updated package versions.


    :rtype: :data:`None`

    """
    n = group.attrs['n']
    n_max = len(group['date'])

    # Increase the size of the datasets by *n_step*
    if n >= n_max:
        if group.attrs['clear_when_full']:
            group = reset_hdf5_log(group, version_values)
            n = 0
        else:
            n_max += group.attrs['n_step']
            group['date'].resize(n_max, axis=0)
            group['version'].resize(n_max, axis=0)
            group['index'].resize(n_max, axis=0)
            group['message'].resize(n_max, axis=0)

    # Parse the passed **idx**
    idx = np.atleast_1d(np.asarray(index))
    generic = idx.dtype.type
    if idx.ndim > 1:
        raise ValueError("The dimensionality of 'index' should be <= 1; "
                         f"observed dimensionality: {idx.ndim!r}")

    if issubclass(generic, np.bool_):
        idx, *_ = idx.nonzero()
    elif not issubclass(generic, np.integer):
        raise TypeError("'idx' expected an integer or boolean array; "
                        f"observed dtype: {idx.dtype!r}")

    # Update the datasets
    group['date'][n] = _get_now()
    group['version'][n] = version_values
    group['index'][n] = idx
    if message is not None:
        group['message'][n] = message

    group.attrs['n'] += 1


def reset_hdf5_log(
    group: h5py.Group,
    version_values: Sequence[Tuple[int, int, int]] | np.ndarray = _VERSION,
) -> h5py.Group:
    r"""Clear and reset the passed ``logger`` Group.

    Examples
    --------
    .. testsetup:: python

        >>> import os
        >>> from shutil import copyfile
        >>> from dataCAT.testing_utils import HDF5_READ, HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)
        >>> _ = copyfile(HDF5_READ, hdf5_file)

    .. code:: python

        >>> import h5py
        >>> from dataCAT import reset_hdf5_log

        >>> hdf5_file = str(...)  # doctest: +SKIP

        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     group = f['ligand/logger']
        ...     print('before:')
        ...     print(group.attrs['n'])
        ...
        ...     group = reset_hdf5_log(group)
        ...     print('\nafter:')
        ...     print(group.attrs['n'])
        before:
        2
        <BLANKLINE>
        after:
        0

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    group : :class:`h5py.File` or :class:`h5py.Group`
        The ``logger`` Group.
    version_values : :class:`Sequence[Tuple[int, int, int]]<typing.Sequence>`
        A sequence with 3-tuples representing to-be updated package versions.

    Returns
    -------
    :class:`h5py.Group`
        The newly (re-)created ``"logger"`` group.

    """
    version_names = group['version_names'][:]
    n_entries = group.attrs['n_step']
    clear_when_full = group.attrs['clear_when_full']

    parent = group.parent
    file = group.file
    del file[group.name]

    return create_hdf5_log(parent, n_entries, clear_when_full, version_names, version_values)


def log_to_dataframe(group: h5py.Group) -> pd.DataFrame:
    """Export the log embedded within **file** to a Pandas DataFrame.

    Examples
    --------
    .. testsetup:: python

        >>> from dataCAT.testing_utils import HDF5_READ as hdf5_file

    .. code:: python

        >>> import h5py
        >>> from dataCAT import log_to_dataframe

        >>> hdf5_file = str(...)  # doctest: +SKIP

        >>> with h5py.File(hdf5_file, 'r') as f:
        ...     group = f['ligand/logger']
        ...     df = log_to_dataframe(group)

        >>> print(df.shape)
        (2, 11)

        >>> print(df[['CAT', 'message', 'index']])  # doctest: +NORMALIZE_WHITESPACE
                                     CAT             message               index
                                   major minor micro
        date
        2020-06-24 15:28:09.861074     0     9     6  update                 [0]
        2020-06-24 15:56:18.971201     0     9     6  append  [1, 2, 3, 4, 5, 6]

    Parameters
    ----------
    group : :class:`h5py.Group`
        The ``logger`` Group.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame containing the content of :code:`file["logger"]`.

    """  # noqa: E501
    n = group.attrs['n']

    # Prepare the columns
    _columns = group['version_names'][:].astype(str)
    columns = pd.MultiIndex.from_product([_columns, group['version'].dtype.names])

    # In case the datasets are empty
    if not n:
        index = pd.Index([], dtype='datetime64[ns]', name='date')
        df = pd.DataFrame(columns=columns, index=index, dtype='int8')
        df[('message', '')] = np.array([], dtype=str)
        df[('index', '')] = np.array([], dtype=object)
        return df

    # Prepare the index
    date = group['date'][:n]
    _index = np.fromiter((datetime(*i) for i in date), count=len(date), dtype='datetime64[us]')
    index = pd.Index(_index, dtype='datetime64[ns]', name='date')

    # Construct and return the DataFrame
    data = group['version'][:n].view('int8')
    df = pd.DataFrame(data, index=index, columns=columns)
    df[('message', '')] = group['message'][:n].astype(str)
    df[('index', '')] = group['index'][:n]
    return df
