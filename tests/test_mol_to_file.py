"""Tests for :mod:`CAT.data_handling.mol_to_file`."""

from os import mkdir
from os.path import (join, isdir, isfile)
from pathlib import Path
from shutil import rmtree

from scm.plams import Molecule
from assertionlib import assertion
from nanoutils import delete_finally

from CAT.data_handling.mol_to_file import mol_to_file, shorten_filename, MAX_FILENAME_BYTES

_PATH = join('tests', 'test_files')
PATH = join(_PATH, 'mol_to_file')
MOL = Molecule(join(_PATH, 'Methanol.pdb'))


@delete_finally(PATH)
def test_mol_to_file() -> None:
    """Test :func:`CAT.data_handling.mol_to_file.mol_to_file`."""
    if not isdir(PATH):
        mkdir(PATH)

    _ref = ('mol.xyz', 'mol.pdb', 'mol.mol', 'mol.mol2')
    ref = [join(PATH, item) for item in _ref]

    mol = MOL.copy()
    mol.properties.name = 'mol'

    mol_list = [mol]
    mol_format = ('xyz', 'pdb', 'mol', 'mol2')
    kwargs = {'path': PATH, 'mol_format': mol_format}

    mol_to_file(mol_list, **kwargs)
    for file in ref:
        assert isfile(file)

    rmtree(PATH)
    mkdir(PATH)
    kwargs['mol_format'] = kwargs['mol_format'][0:2]
    mol_to_file(mol_list, **kwargs)
    for file in ref[:2]:
        assert isfile(file)
    for file in ref[2:]:
        assert not isfile(file)

    kwargs['path'] = join(PATH, 'bob')
    assertion.assert_(mol_to_file, mol_list, exception=FileNotFoundError, **kwargs)

    kwargs['path'] = join(PATH, 'mol.xyz')
    assertion.assert_(mol_to_file, mol_list, exception=NotADirectoryError, **kwargs)

    kwargs['path'] = PATH
    kwargs['mol_format'] = ('bob')
    assertion.assert_(mol_to_file, mol_list, exception=ValueError, **kwargs)


def test_mol_to_file_overwrite(tmp_path: Path) -> None:
    """Test that ``overwrite=False`` leaves existing files untouched."""
    mol = MOL.copy()
    mol.properties.name = 'mol'
    xyz = tmp_path / 'mol.xyz'
    pdb = tmp_path / 'mol.pdb'
    xyz.write_text('old')

    mol_to_file([mol], path=str(tmp_path), overwrite=False, mol_format=('xyz', 'pdb'))
    assert xyz.read_text() == 'old'
    assert pdb.is_file()

    mol_to_file([mol], path=str(tmp_path), overwrite=True, mol_format=('xyz', 'pdb'))
    assert xyz.read_text() != 'old'


def test_mol_to_file_long_name(tmp_path: Path) -> None:
    """Test that molecules with too-long names are written under a shortened name."""
    names = ['C' * 300, 'C' * 299 + 'O']
    mol_list = []
    for name in names:
        mol = MOL.copy()
        mol.properties.name = name
        mol_list.append(mol)

    mol_to_file(mol_list, path=str(tmp_path), mol_format=('xyz', 'mol2'))
    files = sorted(f.name for f in tmp_path.iterdir())
    ref = sorted(f'{shorten_filename(n, 4)}.{ext}' for n in names for ext in ('xyz', 'mol2'))
    assert files == ref
    assert all(len(f.encode()) <= MAX_FILENAME_BYTES for f in files)


def test_shorten_filename() -> None:
    """Test :func:`CAT.data_handling.mol_to_file.shorten_filename`."""
    name = 'C' * (MAX_FILENAME_BYTES - 4)
    assert shorten_filename(name, ext_len=3) == name
    assert shorten_filename(name + 'C', ext_len=3) != name + 'C'
    assert len(shorten_filename('C' * 1000, ext_len=3)) == MAX_FILENAME_BYTES - 4
    assert shorten_filename('C' * 1000) != shorten_filename('C' * 999 + 'O')
