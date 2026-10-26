"""Tests for :mod:`dataCAT.context_managers`."""

import sys
import copy
import pickle
from os.path import join
from functools import partial

import numpy as np
import pandas as pd

from assertionlib import assertion

from dataCAT import OpenLig, OpenQD

PATH = join('tests', 'test_files')


def test_filemanagerabc() -> None:
    """Test :class:`dataCAT.context_managers.FileManagerABC`."""
    file1 = join(PATH, 'qd.csv')
    file2 = join(PATH, 'qd.csv')
    file3 = join(PATH, 'ligand.csv')
    obj1 = OpenQD(file1, write=False)
    obj2 = OpenQD(file2, write=False)
    obj3 = OpenQD(file3, write=False)

    assertion.eq(obj1, obj2)
    assertion.eq(hash(obj1), hash(obj2))
    assertion.ne(obj1, obj3)
    assertion.ne(obj1, 1)

    obj1_str = repr(obj1)
    assertion.contains(obj1_str, obj1.__class__.__name__)
    assertion.contains(obj1_str, str(obj1.write))
    if sys.platform != 'win32':
        assertion.contains(obj1_str, str(obj1.filename))

    assertion.is_(copy.copy(obj1), obj1)
    assertion.is_(copy.deepcopy(obj1), obj1)

    dump = pickle.dumps(obj1)
    load = pickle.loads(dump)
    assertion.eq(load, obj1)


def test_openlig() -> None:
    """Test :class:`dataCAT.context_managers.OpenLig`."""
    path = join(PATH, 'ligand.csv')
    manager = partial(OpenLig, path)

    idx = pd.MultiIndex.from_tuples(
        [('-', '-'), ('C[O-]', 'O2'), ('CC[O-]', 'O3'), ('CCC[O-]', 'O4')],
        names=['smiles', 'anchor']
    )

    columns = pd.MultiIndex.from_tuples(
        [('formula', ''), ('hdf5 index', ''), ('opt', ''), ('settings', '1')],
        names=['index', 'sub-index']
    )

    data = np.array(
        [['str', -1, False, 'str'],
         ['C1H3O1', 2, True, 'RDKit_2019.03.2 0'],
         ['C2H5O1', 1, True, 'RDKit_2019.03.2 0'],
         ['C3H7O1', 0, True, 'RDKit_2019.03.2 0']], dtype=object
    )

    ref = pd.DataFrame(data, idx, columns)
    ref['opt'] = ref['opt'].astype(bool)
    ref['hdf5 index'] = ref['hdf5 index'].astype(int)

    with manager(write=False) as df:
        for k in ref:
            np.testing.assert_array_equal(df[k], ref[k])
        np.testing.assert_array_equal(df.index, ref.index)
        np.testing.assert_array_equal(df.columns, ref.columns)


def test_openqd() -> None:
    """Test :class:`dataCAT.context_managers.OpenQD`."""
    path = join(PATH, 'qd.csv')
    manager = partial(OpenQD, path)

    idx = pd.MultiIndex.from_tuples(
        [('-', '-', '-', '-'),
         ('Cd68Cl26Se55', '124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149', 'CCC[O-]', 'O4'),  # noqa
         ('Cd68Cl26Se55', '124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149', 'CC[O-]', 'O3'),  # noqa
         ('Cd68Cl26Se55', '124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149', 'C[O-]', 'O2')],  # noqa
        names=['core', 'core anchor', 'ligand smiles', 'ligand anchor']
    )

    columns = pd.MultiIndex.from_tuples(
        [('hdf5 index', ''), ('ligand count', ''), ('opt', ''), ('settings', '1'),
         ('settings', '2')],
        names=['index', 'sub index']
    )

    data = np.array(
        [[-1, -1, False, 'str', 'str'],
         [0, -1, True, 'type 0', 'type 0'],
         [1, -1, True, 'type 0', 'type 0'],
         [2, -1, True, 'type 0', 'type 0']], dtype=object
    )

    ref = pd.DataFrame(data, idx, columns)
    ref['opt'] = ref['opt'].astype(bool)
    ref['hdf5 index'] = ref['hdf5 index'].astype(int)
    ref['ligand count'] = ref['ligand count'].astype(int)

    with manager(write=False) as df:
        for k in ref:
            np.testing.assert_array_equal(df[k], ref[k])
        np.testing.assert_array_equal(df.index, ref.index)
        np.testing.assert_array_equal(df.columns, ref.columns)
