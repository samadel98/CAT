"""A module for storing quantum mechanical properties in hdf5 format.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    create_prop_group
    create_prop_dset
    update_prop_dset
    validate_prop_group
    index_to_pandas
    prop_to_dataframe

API
---
.. autofunction:: create_prop_group
.. autofunction:: create_prop_dset
.. autofunction:: update_prop_dset
.. autofunction:: validate_prop_group
.. autofunction:: index_to_pandas
.. autofunction:: prop_to_dataframe

"""

from __future__ import annotations

from typing import Union, Sequence, Any, Optional, Dict, TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from assertionlib import assertion

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
else:
    DTypeLike = 'numpy.typing.DTypeLike'

__all__ = ['create_prop_group', 'create_prop_dset', 'update_prop_dset',
           'validate_prop_group', 'prop_to_dataframe', 'index_to_pandas']

PROPERTY_DOC = r"""A h5py Group containing an arbitrary number of quantum-mechanical properties.

Attributes
----------
\*args : dataset
    An arbitrary user-specified property-containing dataset.

index : attribute
    A reference to the dataset used as dimensional scale for all property
    datasets embedded within this group.

"""


def create_prop_group(file: h5py.Group, scale: h5py.Dataset) -> h5py.Group:
    r"""Create a group for holding user-specified properties.

    .. testsetup:: python

        >>> import os
        >>> from shutil import copyfile
        >>> from dataCAT.testing_utils import HDF5_READ, HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)
        >>> _ = copyfile(HDF5_READ, hdf5_file)

    .. code:: python

        >>> import h5py
        >>> from dataCAT import create_prop_group

        >>> hdf5_file = str(...)  # doctest: +SKIP
        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     scale = f.create_dataset('index', data=np.arange(10))
        ...     scale.make_scale('index')
        ...
        ...     group = create_prop_group(f, scale=scale)
        ...     print('group', '=', group)
        group = <HDF5 group "/properties" (0 members)>

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    file : :class:`h5py.File` or :class:`h5py.Group`
        The File or Group where the new ``"properties"`` group should be created.
    scale : :class:`h5py.DataSet`
        The dimensional scale which will be attached to all property datasets
        created by :func:`dataCAT.create_prop_dset`.

    Returns
    -------
    :class:`h5py.Group`
        The newly created group.

    """
    # Construct the group
    grp = file.create_group('properties', track_order=True)
    grp.attrs['index'] = scale.ref
    grp.attrs['__doc__'] = np.bytes_(PROPERTY_DOC)
    return grp


def create_prop_dset(group: h5py.Group, name: str, dtype: DTypeLike | None = None,
                     prop_names: Optional[Sequence[str]] = None,
                     **kwargs: Any) -> h5py.Dataset:
    r"""Construct a new dataset for holding a user-defined molecular property.

    Examples
    --------
    In the example below a new dataset is created for storing
    solvation energies in water, methanol and ethanol.

    .. testsetup:: python

        >>> import os
        >>> from shutil import copyfile
        >>> from dataCAT.testing_utils import HDF5_READ, HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)
        >>> _ = copyfile(HDF5_READ, hdf5_file)

        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     scale = f.create_dataset('index', data=np.arange(10))
        ...     scale.make_scale('index')
        ...     _ = create_prop_group(f, scale=scale)

    .. code:: python

        >>> import h5py
        >>> from dataCAT import create_prop_dset

        >>> hdf5_file = str(...)  # doctest: +SKIP

        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     group = f['properties']
        ...     prop_names = ['water', 'methanol', 'ethanol']
        ...
        ...     dset = create_prop_dset(group, 'E_solv', prop_names=prop_names)
        ...     dset_names = group['E_solv_names']
        ...
        ...     print('group', '=', group)
        ...     print('group["E_solv"]', '=', dset)
        ...     print('group["E_solv_names"]', '=', dset_names)
        group = <HDF5 group "/properties" (2 members)>
        group["E_solv"] = <HDF5 dataset "E_solv": shape (10, 3), type "<f4">
        group["E_solv_names"] = <HDF5 dataset "E_solv_names": shape (3,), type "|S8">

    .. testcleanup:: python

        >>> import os

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    group : :class:`h5py.Group`
        The ``"properties"`` group where the new dataset will be created.
    name : :class:`str`
        The name of the new dataset.
    prop_names : :class:`Sequence[str]<typing.Sequence>`, optional
        The names of each row in the to-be created dataset.
        Used for defining the length of the second axis and
        will be used as a dimensional scale for aforementioned axis.
        If :data:`None`, create a 1D dataset (with no columns) instead.
    dtype : dtype-like
        The data type of the to-be created dataset.
        Defaults to :class:`numpy.float32` if :data:`None`.
    \**kwargs : :data:`~Any`
        Further keyword arguments for the h5py :meth:`~h5py.Group.create_dataset` method.

    Returns
    -------
    :class:`h5py.Dataset`
        The newly created dataset.

    """
    if dtype is None:
        dtype = np.float32

    scale_name = f'{name}_names'
    index_ref = group.attrs['index']
    index = group.file[index_ref]
    index_name = index.name.rsplit('/', 1)[-1]
    n = len(index)

    # If no prop_names are specified
    if prop_names is None:
        dset = group.create_dataset(name, shape=(n,), maxshape=(None,), dtype=dtype, **kwargs)
        dset.dims[0].label = index_name
        dset.dims[0].attach_scale(index)
        return dset

    # Parse the names
    name_array = np.asarray(prop_names, dtype=np.bytes_)
    if name_array.ndim != 1:
        raise ValueError("'prop_names' expected None or a 1D array-like object; "
                         f"observed dimensionality: {name_array.ndim!r}")

    # Construct the new datasets
    m = len(name_array)
    dset = group.create_dataset(
        name,
        shape=(n, m),
        maxshape=(None, m),
        dtype=dtype,
        fillvalue=(_null_value(dtype) if dtype != object else None),
        **kwargs
    )
    scale = group.create_dataset(scale_name, data=name_array, shape=(m,), dtype=name_array.dtype)
    scale.make_scale(scale_name)

    # Set the dimensional scale
    dset.dims[0].label = index_name
    dset.dims[0].attach_scale(index)
    dset.dims[1].label = scale_name
    dset.dims[1].attach_scale(scale)
    return dset


def _null_value(dtype_like: DTypeLike) -> np.generic:
    dtype = np.dtype(dtype_like)
    generic: type[np.generic] = dtype.type

    if issubclass(generic, (np.number, np.bool_)):  # Numerical scalars
        return generic(False)
    elif not issubclass(generic, np.void):  # Strings, bytes & datetime64
        return generic('')

    # Structured dtypes
    values = (v[0] for v in dtype.fields.values())  # type: ignore[union-attr]
    data = tuple(_null_value(field_dtype) for field_dtype in values)
    return np.array(data, dtype=dtype).take(0)  # type: ignore[no-any-return]


def _resize_prop_dset(dset: h5py.Dataset) -> None:
    """Ensure that **dset** is as long as its dimensional scale."""
    scale = dset.dims[0][0]
    n = len(scale)
    if n > len(dset):
        dset.resize(n, axis=0)


def update_prop_dset(dset: h5py.Dataset, data: np.ndarray,
                     index: Union[None, slice, np.ndarray] = None) -> None:
    """Update **dset** at position **index** with **data**.

    Parameters
    ----------
    dset : :class:`h5py.Dataset`
        The to-be updated h5py dataset.
    data : :class:`numpy.ndarray`
        An array containing the to-be added data.
    index : :class:`slice` or :class:`numpy.ndarray`, optional
        The indices of all to-be updated elements in **dset**.
        **index** either should be of the same length as **data**.


    :rtype: :data:`None`

    """
    idx = slice(None) if index is None else index

    try:
        _resize_prop_dset(dset)
        dset[idx] = data
    except Exception:
        validate_prop_group(dset.parent)
        raise


def validate_prop_group(group: h5py.Group) -> None:
    """Validate the passed hdf5 **group**, ensuring it is compatible with :func:`create_prop_group` and :func:`create_prop_group`.

    This method is called automatically when an exception is raised by :func:`update_prop_dset`.

    Parameters
    ----------
    group : :class:`h5py.Group`
        The to-be validated hdf5 Group.

    Raises
    ------
    :exc:`AssertionError`
        Raised if the validation process fails.

    """  # noqa: E501
    assertion.isinstance(group, h5py.Group)

    idx_ref = group.attrs['index']
    idx = group.file[idx_ref]

    iterator = ((k, v) for k, v in group.items() if not k.endswith('_names'))
    for name, dset in iterator:
        assertion.le(len(dset), len(idx), message=f'{name!r} invalid dataset length')
        assertion.contains(dset.dims[0].keys(), 'index', message=f'{name!r} missing dataset scale')
        assertion.eq(dset.dims[0]['index'], idx, message=f'{name!r} invalid dataset scale')


def prop_to_dataframe(dset: h5py.Dataset, dtype: DTypeLike | None = None) -> pd.DataFrame:
    """Convert the passed property Dataset into a DataFrame.

    Examples
    --------
    .. testsetup:: python

        >>> from dataCAT.testing_utils import HDF5_READ as hdf5_file

    .. code:: python

        >>> import h5py
        >>> from dataCAT import prop_to_dataframe

        >>> hdf5_file = str(...)  # doctest: +SKIP

        >>> with h5py.File(hdf5_file, 'r') as f:
        ...     dset = f['ligand/properties/E_solv']
        ...     df = prop_to_dataframe(dset)
        ...     print(df)  # doctest: +NORMALIZE_WHITESPACE
        E_solv_names             water  methanol   ethanol
        ligand ligand anchor
        O=C=O  O1            -0.918837 -0.151129 -0.177396
               O3            -0.221182 -0.261591 -0.712906
        CCCO   O4            -0.314799 -0.784353 -0.190898

    Parameters
    ----------
    dset : :class:`h5py.Dataset`
        The property-containing Dataset of interest.
    dtype : dtype-like, optional
        The data type of the to-be returned DataFrame.
        Use :data:`None` to default to the data type of **dset**.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame constructed from the passed **dset**.

    """  # noqa: E501
    # Construct the index
    dim0 = dset.dims[0]
    scale0 = dim0[0]
    index = index_to_pandas(scale0)

    # Construct the columns
    if dset.ndim == 1:
        full_name = dset.name
        name = full_name.rsplit('/', 1)[-1]
        columns = pd.Index([name])

    else:
        dim1 = dset.dims[1]
        scale1 = dim1[0]
        columns = pd.Index(scale1[:].astype(str), name=dim1.label)

    # Create and return the dataframe
    if dtype is None:
        return pd.DataFrame(dset[:], index=index, columns=columns)

    # If possible, let h5py handle the datatype conversion
    # This will often fail when dset.dtype consists of variable-length bytes-strings
    try:
        return pd.DataFrame(dset.astype(dtype)[:], index=index, columns=columns)
    except (ValueError, TypeError):
        return pd.DataFrame(dset[:].astype(dtype), index=index, columns=columns)


def index_to_pandas(dset: h5py.Dataset, fields: None | Sequence[str] = None) -> pd.MultiIndex:
    """Construct an MultiIndex from the passed ``index`` dataset.

    Examples
    --------
    .. testsetup:: python

        >>> from dataCAT.testing_utils import HDF5_READ as filename

    .. code:: python

        >>> from dataCAT import index_to_pandas
        >>> import h5py

        >>> filename = str(...)  # doctest: +SKIP

        # Convert the entire dataset
        >>> with h5py.File(filename, "r") as f:
        ...     dset: h5py.Dataset = f["ligand"]["index"]
        ...     index_to_pandas(dset)
        MultiIndex([('O=C=O', 'O1'),
                    ('O=C=O', 'O3'),
                    ( 'CCCO', 'O4')],
                   names=['ligand', 'ligand anchor'])

        # Convert a subset of fields
        >>> with h5py.File(filename, "r") as f:
        ...     dset = f["ligand"]["index"]
        ...     index_to_pandas(dset, fields=["ligand"])
        MultiIndex([('O=C=O',),
                    ('O=C=O',),
                    ( 'CCCO',)],
                   names=['ligand'])

    Parameters
    ----------
    dset : :class:`h5py.Dataset`
        The relevant ``index`` dataset.
    fields : :class:`Sequence[str]<collections.abc.Sequence>`
        The names of the ``index`` fields that are to-be included in the
        returned MultiIndex. If :data:`None`, include all fields.

    Returns
    -------
    :class:`pandas.MultiIndex`
        A multi-index constructed from the passed dataset.

    """
    # Fast-path for non-void-based datasets
    if dset.dtype.fields is None:
        if h5py.check_string_dtype(dset.dtype):
            ar = dset[:].astype(str)
        elif h5py.check_vlen_dtype(dset.dtype):
            ar = _vlen_to_tuples(dset[:])
        else:
            ar = dset[:]
        return pd.MultiIndex.from_arrays([ar])

    # Parse the `fields` parameter
    if fields is None:
        field_names = list(dset.dtype.fields.keys())
        iterator = ((name, f_dtype) for name, (f_dtype, *_) in dset.dtype.fields.items())
    else:
        field_names = list(fields)
        iterator = ((name, dset.dtype.fields[name][0]) for name in fields)
    if len(field_names) == 0:
        raise ValueError("At least one field is required")

    fields_lst = []
    index_ar = dset[:]
    for name, field_dtype in iterator:
        # It's a bytes-string; decode it
        if h5py.check_string_dtype(field_dtype):
            ar = index_ar[name].astype(str)

        # It's a h5py `vlen` dtype; convert it into a list of tuples
        elif h5py.check_vlen_dtype(field_dtype):
            ar = _vlen_to_tuples(index_ar[name])

        else:
            ar = index_ar[name]
        fields_lst.append(ar)
    return pd.MultiIndex.from_arrays(fields_lst, names=field_names)


def _vlen_to_tuples(array: np.ndarray) -> np.ndarray:
    """Convert an (object) array consisting of arrays into an (object) array of tuples."""
    cache: Dict[bytes, tuple] = {}
    ret = np.empty_like(array, dtype=object)

    for i, ar in enumerate(array):
        byte = ar.tobytes()
        try:
            tup = cache[byte]
        except KeyError:
            cache[byte] = tup = tuple(ar)
        ret[i] = tup
    return ret
