"""A module with testing functions for **dataCAT**.

Index
-----
.. currentmodule:: dataCAT.testing_utils
.. autosummary::
    MOL_TUPLE
    MOL
    PDB
    HDF5_TMP
    HDF5_READ

API
---
.. autodata:: MOL_TUPLE
    :annotation: : Tuple[Molecule, ...] = ...
.. autodata:: MOL
    :annotation: : Molecule = ...
.. autodata:: PDB
    :annotation: : PDBContainer = ...
.. autodata:: HDF5_TMP
    :annotation: : pathlib.Path = ...
.. autodata:: HDF5_READ
    :annotation: : pathlib.Path = ...

"""

import os
from pathlib import Path

from scm.plams import readpdb, Molecule

from .pdb_array import PDBContainer
from .data import PDB_TUPLE

__all__ = ['MOL_TUPLE', 'MOL', 'PDB', 'HDF5_TMP', 'HDF5_READ']


def _readpdb(path: str | os.PathLike[str]) -> Molecule:
    """Read a .pdb file, ensuring that the file is closed afterwards."""
    with open(path) as f:
        return readpdb(f)


#: A tuple of PLAMS Molecules.
MOL_TUPLE: tuple[Molecule, ...] = tuple(_readpdb(f) for f in PDB_TUPLE)

#: A PLAMS Molecule.
MOL: Molecule = MOL_TUPLE[0]

#: A PDBContainer.
PDB: PDBContainer = PDBContainer.from_molecules(MOL_TUPLE)

#: A path to a temporary (to-be created) hdf5 file.
HDF5_TMP = Path('tests') / 'test_files' / '.structures.hdf5'

#: A path to a read-only hdf5 file.
HDF5_READ = Path('tests') / 'test_files' / 'database' / 'structures.hdf5'
