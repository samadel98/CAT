"""Tests for :mod:`nanoCAT.bulk.mol_graph2`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scm.plams import Molecule, Settings, from_smiles

from CAT.base import prep
from CAT.workflows import BRANCH_DISTANCE, BRANCH_SIZE
from nanoCAT.bulk.mol_graph2 import get_branching_graph

PATH = Path("tests") / "test_files"

SETTINGS = Settings({
    "input_cores": [str((PATH / "core" / "Cd68Se55.xyz").resolve())],
    "input_ligands": ["OC(C)=O", "OC(=O)c1ccccc1", "OC(=O)CC(CC)CCCC"],
    "optional": {
        "core": {"dummy": "Cl"},
        "ligand": {"split": True, "branch_distance": True},
        "qd": {"construct_qd": False},
    },
})


def _state(mol: Molecule) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    return [at.symbol for at in mol], mol.as_array(), [mol.index(b) for b in mol.bonds]


@pytest.mark.parametrize("smiles,distance,size", [
    ("[O-]C(=O)CC(C)C", [0], [16]),
    ("[O-]C(=O)c1ccccc1", [0, 1], [3, 11]),
    ("OCC(CC)CCCC", [0, 4], [22, 5]),
])
def test_get_branching_graph(smiles: str, distance: list[int], size: list[int]) -> None:
    """Test :func:`get_branching_graph`; it must not modify the molecule."""
    mol = from_smiles(smiles)
    ref = _state(mol)

    graph = get_branching_graph(mol, mol[1])
    assert list(graph.branch_distance) == distance
    assert list(graph.branch_size) == size
    assert sorted(np.concatenate(graph.branches).tolist()) == list(range(len(mol)))

    symbols, xyz, bonds = _state(mol)
    assert symbols == ref[0] and bonds == ref[2]
    np.testing.assert_array_equal(xyz, ref[1])


def test_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The workflow fills both columns, keeps the ligands as they are and stores the results."""
    monkeypatch.chdir(tmp_path)
    settings = SETTINGS.copy()
    settings.path = str(tmp_path)

    s = settings.copy()
    s.optional.ligand.branch_distance = False
    *_, ligand_df_ref = prep(s)

    *_, ligand_df1 = prep(settings.copy())
    *_, ligand_df2 = prep(settings.copy())  # Read from the database
    for ligand_df in (ligand_df1, ligand_df2):
        branches = ligand_df[[BRANCH_DISTANCE, BRANCH_SIZE]].astype(str).values.tolist()
        assert branches == [["0", "7"], ["0-1", "11-3"], ["0-3", "21-7"]]

    # The ligands come from the database (single precision) in both runs
    for mol_ref, mol in zip(ligand_df_ref["mol"], ligand_df1["mol"]):
        np.testing.assert_allclose(mol.as_array(), mol_ref.as_array(), atol=1e-4)
