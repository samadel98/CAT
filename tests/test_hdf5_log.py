"""Tests for :mod:`dataCAT.hdf5_log`."""

from types import MappingProxyType
from shutil import copyfile

import h5py
import numpy as np
import pandas as pd

from assertionlib import assertion
from nanoutils import delete_finally
from dataCAT import update_hdf5_log, reset_hdf5_log, log_to_dataframe
from dataCAT.testing_utils import HDF5_READ, HDF5_TMP


@delete_finally(HDF5_TMP)
def test_update_hdf5_log1() -> None:
    """Test :func:`dataCAT.update_hdf5_log`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+', libver='latest') as f:
        group = f['ligand/logger']

        i = len(group['date'])
        n0 = group.attrs['n']
        assertion.truth(n0)

        i += group.attrs['n_step']
        group.attrs['n'] = 100
        update_hdf5_log(group, index=[0])

        n1 = group.attrs['n']
        assertion.eq(n1, 101)
        for name, dset in group.items():
            if name != 'version_names':
                assertion.len_eq(dset, i)


@delete_finally(HDF5_TMP)
def test_update_hdf5_log2() -> None:
    """Test :func:`dataCAT.update_hdf5_log`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+', libver='latest') as f:
        group = f['ligand/logger']
        group.attrs['clear_when_full'] = True

        i = len(group['date'])
        n0 = group.attrs['n']
        assertion.truth(n0)

        group.attrs['n'] = 100
        update_hdf5_log(group, index=[0])
        group_new = f['ligand/logger']

        n1 = group_new.attrs['n']
        assertion.eq(n1, 1)
        for name, dset in group_new.items():
            if name != 'version_names':
                assertion.len_eq(dset, i)


REF_COLUMNS = MappingProxyType({
    ('CAT', 'major'): np.dtype('int8'),
    ('CAT', 'minor'): np.dtype('int8'),
    ('CAT', 'micro'): np.dtype('int8'),
    ('Nano-CAT', 'major'): np.dtype('int8'),
    ('Nano-CAT', 'minor'): np.dtype('int8'),
    ('Nano-CAT', 'micro'): np.dtype('int8'),
    ('Data-CAT', 'major'): np.dtype('int8'),
    ('Data-CAT', 'minor'): np.dtype('int8'),
    ('Data-CAT', 'micro'): np.dtype('int8'),
    # The default string dtype: `object` for pandas < 3.0 and `str` for pandas >= 3.0
    ('message', ''): pd.Series(['a']).dtype,
    ('index', ''): np.dtype('O')
})


@delete_finally(HDF5_TMP)
def test_log_to_dataframe() -> None:
    """Test :func:`dataCAT.log_to_dataframe`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+', libver='latest') as f:
        group = f['ligand/logger']
        reset_hdf5_log(group)
        df = log_to_dataframe(group)
        dct = {k: v.dtype for k, v in df.items()}
        assertion.eq(dct, REF_COLUMNS)
