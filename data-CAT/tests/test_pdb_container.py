"""Tests for :class:`dataCAT.PDBContainer`."""

import os
import copy
import pickle
from shutil import copyfile
from pathlib import Path

import h5py
import numpy as np

from scm.plams import writepdb
from assertionlib import assertion
from nanoutils import delete_finally
from dataCAT import PDBContainer
from dataCAT.testing_utils import PDB, HDF5_READ, HDF5_TMP

PDB_OUTPUT = Path('tests') / 'test_files' / '.pdb_files'


def test_pickle() -> None:
    """Test :meth:`PDBContainer.__reduce__`."""
    dumps = pickle.dumps(PDB)
    loads = pickle.loads(dumps)
    assertion.eq(loads, PDB)


def test_eq() -> None:
    """Test :meth:`PDBContainer.__eq__`."""
    pdb1 = PDB[:]
    pdb2 = PDB[0]

    assertion.eq(pdb1, PDB)
    assertion.ne(pdb2, PDB)
    assertion.ne(1, PDB)

    # We're gona cheat here by manually specifying the object's hash
    pdb3 = pdb2[:]
    pdb3._hash = PDB._hash
    assertion.ne(pdb3, PDB)


def test_hash() -> None:
    """Test :meth:`PDBContainer.__hash__`."""
    pdb = PDB[:]
    assertion.eq(hash(pdb), hash(PDB))


def test_copy() -> None:
    """Test :meth:`PDBContainer.__copy__` and :meth:`PDBContainer.__deepcopy__`."""
    pdb1 = copy.copy(PDB)
    pdb2 = copy.deepcopy(PDB)

    assertion.is_(PDB, pdb1)
    assertion.is_(PDB, pdb2)


def test_init() -> None:
    """Test :meth:`PDBContainer.__init__`."""
    atoms = [(True, 1, 'Cad', 'LIG', 'A', 1, -5.231, 0.808, -0.649, 1, 0, 'C', 0, 0)]
    bonds = [(1, 2, 1)]
    pdb1 = PDBContainer(atoms, bonds, 1, 1)
    np.testing.assert_array_equal(pdb1.scale, [0])
    assertion.eq(pdb1.atoms.shape, (1, 1))
    assertion.eq(pdb1.bonds.shape, (1, 1))
    assertion.isinstance(pdb1.atoms, np.recarray)

    index = range(5, 6)
    pdb2 = PDBContainer(atoms, bonds, 1, 1, index)
    np.testing.assert_array_equal(pdb2.scale, [5])

    atom_count = np.array([1], dtype=np.int32)
    pdb3 = PDBContainer(atoms, bonds, atom_count, 1, copy=False)
    assertion.truth(np.shares_memory(pdb3.atom_count, atom_count))
    assertion.is_(pdb3.atom_count.flags.writeable, False)

    pdb4 = PDBContainer(atoms, bonds, atom_count.copy(), 1, copy=True)
    assertion.is_(np.shares_memory(pdb4.atom_count, atom_count), False)


@delete_finally(PDB_OUTPUT)
def test_to_molecules() -> None:
    """Test :meth:`PDBContainer.to_molecules`."""
    os.mkdir(PDB_OUTPUT)
    mol_list = PDB.to_molecules()
    for i, mol in enumerate(mol_list):
        with open(PDB_OUTPUT / f'mol{i}.pdb', 'w') as f:
            writepdb(mol, f)


@delete_finally(HDF5_TMP)
def test_validate() -> None:
    """Test :meth:`PDBContainer.validate_hdf5`."""
    with h5py.File(HDF5_TMP, 'a', libver='latest') as f:
        group = f.create_group('test')

        try:
            PDB.from_hdf5(group)
        except AssertionError as ex:
            assertion.isinstance(ex.__context__, KeyError)
            dset = group.create_dataset('atoms', data=[1])
        else:
            raise AssertionError("Failed to raise an AssertionError")

        try:
            PDB.to_hdf5(group, None)
        except AssertionError as ex:
            assertion.isinstance(ex.__context__, RuntimeError)
            scale = group.create_dataset('index', data=[1], maxshape=(None,))
            scale.make_scale('index')
            dset.dims[0].attach_scale(scale)
        else:
            raise AssertionError("Failed to raise an AssertionError")

        try:
            PDB.to_hdf5(group, None)
        except AssertionError as ex:
            assertion.isinstance(ex.__context__, IndexError)
        else:
            raise AssertionError("Failed to raise an AssertionError")


@delete_finally(HDF5_TMP)
def test_index_other() -> None:
    """Test :func:`dataCAT.update_hdf5_log`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+') as f:
        scale = f['ligand/index']
        group = PDBContainer.create_hdf5_group(f, 'test', scale=scale)
        dset = group['atoms']
        assertion.eq(scale, dset.dims[0]['index'])
