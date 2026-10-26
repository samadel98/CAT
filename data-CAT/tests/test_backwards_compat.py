"""Test the backwards compatbility of data-CAT ``>= 0.3`` with respect to ``0.2.3``."""

import shutil
from pathlib import Path

import numpy as np

from nanoutils import delete_finally
from assertionlib import assertion
from CAT.utils import get_formula
from dataCAT import Database, PDBContainer
from dataCAT.dtype import LIG_IDX_DTYPE
from dataCAT.testing_utils import _readpdb as readpdb

PATH = Path('tests') / 'test_files'

DB_PATH_OLD3 = PATH / 'database_pre_0_3'
DB_PATH_NEW3 = PATH / '.database_pre_0_3'
DB_PATH_OLD4 = PATH / 'database_pre_0_4'
DB_PATH_NEW4 = PATH / '.database_pre_0_4'


@delete_finally(DB_PATH_NEW3)
def test_create_database_3() -> None:
    """Test the backwards compatiblity of pre-``0.3`` databases."""
    shutil.copytree(DB_PATH_OLD3, DB_PATH_NEW3)
    db = Database(DB_PATH_NEW3)

    with db.hdf5('r', libver='latest') as f:
        pdb = PDBContainer.from_hdf5(f['ligand'])
        mol_list = pdb.to_molecules()

    mol_list_ref = []
    files = ('C3H7O1.pdb', 'C2H5O1.pdb', 'C1H3O1.pdb')
    for _path in files:
        path = str(DB_PATH_NEW3 / _path)
        mol_list_ref.append(readpdb(path))

    # Compare molecules
    for mol1, mol2 in zip(mol_list, mol_list_ref):
        msg = f'{get_formula(mol1)} & {get_formula(mol2)}'
        np.testing.assert_allclose(mol1, mol2, rtol=0, atol=10**-2, err_msg=msg)
        np.testing.assert_array_equal(mol1.bond_matrix(), mol2.bond_matrix(), err_msg=msg)

    with db.hdf5('r', libver='latest') as f:
        group = f['ligand']
        pdb.validate_hdf5(group)


@delete_finally(DB_PATH_NEW4)
def test_create_database_4() -> None:
    """Test the backwards compatiblity of pre-``0.4`` databases."""
    shutil.copytree(DB_PATH_OLD4, DB_PATH_NEW4)
    db = Database(DB_PATH_NEW4)

    with db.hdf5('r', libver='latest') as f:
        grp = f['ligand']
        scale = grp['index']
        pdb = PDBContainer.from_hdf5(grp)

        assertion.eq(grp['atoms'].dims[0]['index'], scale)
        assertion.eq(grp['bonds'].dims[0]['index'], scale)
        assertion.eq(grp['atom_count'].dims[0]['index'], scale)
        assertion.eq(grp['bond_count'].dims[0]['index'], scale)

        assertion.eq(grp['atoms'].dims[0].label, 'index')
        assertion.eq(grp['bonds'].dims[0].label, 'index')
        assertion.eq(grp['atom_count'].dims[0].label, 'index')
        assertion.eq(grp['bond_count'].dims[0].label, 'index')

        assertion.eq(grp['atoms'].dims[1].label, 'atoms')
        assertion.eq(grp['bonds'].dims[1].label, 'bonds')
        pdb.validate_hdf5(grp)

        assertion.contains(grp, 'properties')
        formula = grp['properties/formula']
        assertion.len_eq(formula, len(scale))
        assertion.eq(formula.dims[0]['index'], scale)

    ref = np.rec.array(None, dtype=LIG_IDX_DTYPE, shape=(3,))
    ref[:] = b''
    assertion.eq(pdb.scale, ref, post_process=np.all)
