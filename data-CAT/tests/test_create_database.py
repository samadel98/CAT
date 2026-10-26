"""Tests for :mod:`dataCAT.create_database`."""

from os.path import join

import h5py
import numpy as np
import pandas as pd

from assertionlib import assertion
from nanoutils import delete_finally

from dataCAT.create_database import create_csv, create_hdf5

PATH = join('tests', 'test_files')

LIGAND_PATH = join(PATH, 'ligand_database.csv')
QD_PATH = join(PATH, 'qd_database.csv')
HDF5_PATH = join(PATH, 'structures.hdf5')


@delete_finally(LIGAND_PATH, QD_PATH)
def test_create_csv() -> None:
    """Test :func:`dataCAT.create_database.create_csv`."""
    dtype1 = {'hdf5 index': int, 'formula': str, 'settings': str, 'opt': bool}
    dtype2 = {'hdf5 index': int, 'settings': str, 'opt': bool}

    filename1 = create_csv(PATH, 'ligand')
    filename2 = create_csv(PATH, 'qd')
    assertion.eq(filename1, LIGAND_PATH)
    assertion.eq(filename2, QD_PATH)
    df1 = pd.read_csv(LIGAND_PATH, index_col=[0, 1], header=[0, 1], dtype=dtype1)
    df2 = pd.read_csv(QD_PATH, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype2)

    assertion.eq(df1.shape, (1, 4))
    assertion.eq(df2.shape, (1, 5))
    assertion.eq(df1.index.names, ['smiles', 'anchor'])
    assertion.eq(df2.index.names, ['core', 'core anchor', 'ligand smiles', 'ligand anchor'])
    assertion.eq(df1.columns.names, ['index', 'sub index'])
    assertion.eq(df2.columns.names, ['index', 'sub index'])
    assertion.contains(df1.index, ('-', '-'))
    assertion.contains(df2.index, ('-', '-', '-', '-'))

    np.testing.assert_array_equal(
        df1.values, np.array([[-1, 'str', False, 'str']], dtype=object)
    )
    np.testing.assert_array_equal(
        df2.values, np.array([[-1, -1, False, 'str', 'str']], dtype=object)
    )

    assertion.assert_(create_csv, PATH, 'bob', exception=ValueError)


@delete_finally(HDF5_PATH)
def test_create_hdf5() -> None:
    """Test :func:`dataCAT.create_database.create_hdf5`."""
    ref_keys1 = ('qd', 'qd_no_opt', 'core', 'core_no_opt', 'ligand', 'ligand_no_opt')
    ref_keys2 = ('job_settings_BDE', 'job_settings_qd_opt', 'job_settings_crs')

    filename = create_hdf5(PATH)
    assertion.eq(filename, HDF5_PATH)
    with h5py.File(HDF5_PATH, 'r', libver='latest') as f:
        for item in ref_keys1:
            assertion.contains(f.keys(), item)
        for item in ref_keys2:
            assertion.contains(f.keys(), item)
            assertion.eq(f[item].ndim, 3)
