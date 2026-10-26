"""Tests for importing pre-built quantum dots (``input_qd``)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scm.plams import Molecule, Settings

from CAT.base import prep
from CAT._compat import readpdb
from CAT.data_handling.mol_import import read_mol
from CAT.data_handling.validate_mol import validate_mol
from CAT.mol_utils import rdkit_props
from CAT.workflows import MOL

PATH = Path('tests') / 'test_files' / 'core'
ETHOXIDE = PATH / 'Cd68Cl26Se55__26_CC[O-]@O3.pdb'
FORMATE = PATH / 'Cd68Cl26Se55__26_O=C[O-]@O2O3.pdb'


def _import_qd(filename: str | Path, **kwargs: object) -> Molecule:
    """Import a single quantum dot with :func:`read_mol`."""
    args = [{str(filename): Settings(kwargs)}]
    validate_mol(args, 'input_qd', str(PATH))
    mol_list = read_mol(args)
    assert len(mol_list) == 1
    return mol_list[0]


def _as_xyz(filename: Path, tmp_path: Path) -> Path:
    """Write **filename** as a bond-less .xyz file."""
    mol = readpdb(filename)
    ret = tmp_path / f'{filename.stem}.xyz'
    mol.write(str(ret))
    return ret


@pytest.mark.parametrize("fmt", ["pdb", "xyz"])
def test_import_ethoxide(fmt: str, tmp_path: Path) -> None:
    """Import a quantum dot from a .pdb file (with bonds) or an .xyz file (without)."""
    filename = ETHOXIDE if fmt == "pdb" else _as_xyz(ETHOXIDE, tmp_path)
    qd = _import_qd(filename, ligand_smiles='CC[O-]', ligand_anchor='[O-]')

    assert len(qd) == 123 + 26 * 8
    assert Counter(at.symbol for at in qd.atoms[:123]) == {'Cd': 68, 'Se': 55}
    np.testing.assert_array_equal(qd.as_array(), [at.coords for at in qd])
    assert sum(rdkit_props(at).charge for at in qd) == 0
    assert qd.properties.ligand_anchor == 'O3'

    # 26 ligands, each in its own residue, with bonds from the SMILES string only
    residues = Counter(rdkit_props(at).pdb_info.ResidueNumber for at in qd)
    assert residues == {1: 123, **dict.fromkeys(range(2, 28), 8)}
    assert len(qd.bonds) == 26 * 7

    # The anchors are oxygens with a formal charge of -1, next to the core
    anchors = [at for at in qd if at.properties.get('anchor')]
    assert len(anchors) == 26
    assert qd.properties.indices == list(range(1, 124)) + [qd.atoms.index(at) + 1 for at in anchors]
    assert {(at.symbol, rdkit_props(at).charge) for at in anchors} == {('O', -1)}
    core_xyz = np.array([at.coords for at in qd.atoms[:123]])
    for at in anchors:
        assert np.linalg.norm(core_xyz - at.coords, axis=1).min() < 3.0


def test_import_formate_anchor(tmp_path: Path) -> None:
    """The anchor of a carboxylate must be the oxygen closest to the core."""
    qd = _import_qd(_as_xyz(FORMATE, tmp_path), ligand_smiles='O=C[O-]', ligand_anchor='[O-]')
    core_xyz = np.array([at.coords for at in qd.atoms[:123]])

    for i in range(26):
        lig = qd.atoms[123 + 4 * i: 127 + 4 * i]
        o_atoms = [at for at in lig if at.symbol == 'O']
        anchor, = [at for at in o_atoms if at.properties.get('anchor')]
        other, = [at for at in o_atoms if not at.properties.get('anchor')]
        d_anchor = np.linalg.norm(core_xyz - anchor.coords, axis=1).min()
        d_other = np.linalg.norm(core_xyz - other.coords, axis=1).min()
        assert round(d_anchor, 1) <= round(d_other, 1)
        assert rdkit_props(anchor).charge == -1
        assert rdkit_props(other).charge == 0


@pytest.mark.parametrize("kwargs,exc", [
    ({'ligand_anchor': '[O-]'}, ValueError),
    ({'ligand_smiles': 'CC[O-]', 'ligand_anchor': 'N'}, Exception),
    ({'ligand_smiles': 'CCC[O-]', 'ligand_anchor': '[O-]'}, Exception),
])
def test_import_qd_errors(kwargs: dict[str, str], exc: type[Exception]) -> None:
    """Missing arguments, an anchor outside the ligand or a ligand that is not there."""
    with pytest.raises(exc):
        _import_qd(ETHOXIDE, **kwargs)


def test_prep_input_qd(tmp_path: Path) -> None:
    """Run CAT on an imported quantum dot."""
    xyz = _as_xyz(ETHOXIDE, tmp_path)
    s = Settings({
        'path': str(tmp_path),
        'input_qd': [{str(xyz): {'ligand_smiles': 'CC[O-]', 'ligand_anchor': '[O-]'}}],
        'optional': {'database': {'read': False, 'write': False}},
    })
    qd_df, core_df, ligand_df = prep(s)
    assert core_df is None and ligand_df is None
    assert qd_df.index.to_list() == [('Cd68Se55', xyz.stem, 'CC[O-]', 'O3')]
    assert len(qd_df[MOL].iloc[0]) == 123 + 26 * 8
