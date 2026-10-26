"""Tests for :func:`nanoCAT.recipes.ligand_descriptors`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scm.plams import Molecule, Settings

from CAT.base import prep
from CAT.workflows import MOL, V_BULK, BRANCH_DISTANCE, BRANCH_SIZE
from nanoCAT.recipes import ligand_descriptors

PATH = Path("tests") / "test_files"
SMILES = ["OC(C)=O", "OC(=O)CC(C)C", "OC(=O)c1ccccc1"]
INDEX = [("CC(=O)[O-]", "O3"), ("CC(C)CC(=O)[O-]", "O6"), ("O=C([O-])c1ccccc1", "O8")]
DIST = [0.0, 2.0]


def test_vs_cat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The descriptors are those of the CAT workflows for the same ligands."""
    core = str((PATH / "core" / "Cd68Se55.xyz").resolve())
    monkeypatch.chdir(tmp_path)
    s = Settings({
        "path": str(tmp_path),
        "input_cores": [core],
        "input_ligands": SMILES,
        "optional": {
            "database": {"read": False, "write": False},
            "core": {"dummy": "Cl"},
            "ligand": {
                "optimize": True, "split": True, "branch_distance": True,
                "cone_angle": {"distance": DIST},
            },
            "qd": {"bulkiness": {"d": 4.5, "h_lim": 10.0}},
        },
    })
    qd_df, _, ligand_df = prep(s)

    df = ligand_descriptors(SMILES, cone_angle_dist=DIST, diameter=4.5, height_lim=10.0)
    assert df.index.to_list() == INDEX
    assert df.index.names == ["smiles", "anchor"]
    assert df.columns.names == ["index", "sub index"]

    ref = ligand_df.loc[df.index]
    assert np.isfinite(df.select_dtypes("number")).all(axis=None)
    for i in DIST:
        key = ("cone_angle", f"dist={i}")
        np.testing.assert_allclose(df[key], ref[key].astype(float), rtol=1e-6, err_msg=key)
    for key in (BRANCH_DISTANCE, BRANCH_SIZE):
        assert df[key].to_list() == ref[key].astype(str).to_list()

    # Every ligand forms a single quantum dot with the Cd68Se55 core
    V_bulk = qd_df[V_BULK].groupby(level=["ligand smiles", "ligand anchor"]).first()
    V_bulk.index.names = ["smiles", "anchor"]
    np.testing.assert_allclose(df[V_BULK], V_bulk.loc[df.index], rtol=1e-6)


def test_return_mol() -> None:
    """Test ``return_mol=True`` and the default distance."""
    df = ligand_descriptors(SMILES[:2], return_mol=True)
    assert df.columns.to_list() == [
        ("cone_angle", "dist=0.0"), V_BULK, BRANCH_DISTANCE, BRANCH_SIZE, MOL,
    ]
    for (smiles, anchor), mol in df[MOL].items():
        assert isinstance(mol, Molecule)
        assert mol.properties.smiles == smiles
        assert mol.properties.dummies in mol.atoms
        i = mol.atoms.index(mol.properties.dummies) + 1
        assert anchor == f"{mol.properties.dummies.symbol}{i}"


def test_anchor() -> None:
    """Test a custom anchor, several anchors per ligand and ``anchor_condition``."""
    smiles = ["OCCO", "OCCCN", "CCCC"]
    df = ligand_descriptors(smiles, anchor="O[H]")
    assert df.index.to_list() == [("[O-]CCO", "O3"), ("[O-]CCO", "O4"), ("NCCC[O-]", "O5")]

    df2 = ligand_descriptors(smiles, anchor=["O[H]", "N[H]"])
    assert df2.index.to_list() == df.index.to_list() + [("[NH-]CCCO", "N4")]

    df3 = ligand_descriptors(smiles, anchor=["O[H]", "N[H]"], anchor_condition=_one_anchor)
    assert len(df3) == 0
    assert df3.index.names == ["smiles", "anchor"]
    assert len(df3.columns) == 4


def _one_anchor(n: int) -> bool:
    return n == 1


def test_processes() -> None:
    """Worker processes give the same table and molecules."""
    ref = ligand_descriptors(SMILES, return_mol=True)
    df = ligand_descriptors(SMILES, return_mol=True, processes=2)
    pd.testing.assert_frame_equal(df.drop(columns=MOL), ref.drop(columns=MOL))
    for mol, mol_ref in zip(df[MOL], ref[MOL]):
        np.testing.assert_allclose(mol.as_array(), mol_ref.as_array())
        assert mol.properties.dummies in mol.atoms
        assert mol.atoms.index(mol.properties.dummies) == \
            mol_ref.atoms.index(mol_ref.properties.dummies)


def test_raise() -> None:
    """Test invalid input."""
    with pytest.raises(ValueError):
        ligand_descriptors(SMILES, cone_angle_dist=[[1.0]])
