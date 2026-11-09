"""Tests for :mod:`nanoCAT.conformers`."""

from __future__ import annotations

import io
import os
import shutil
import contextlib
import importlib.util
from pathlib import Path
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
from scm.plams import JobManager, Settings, config, from_smiles, init

from CAT.base import prep
from CAT.workflows import CONFORMERS_COUNT, CONFORMERS_E_REL
from nanoCAT.conformers import conformer_search, _unique

requires_tblite = pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="requires tblite"
)

PATH = Path("tests") / "test_files"
WORKDIR = PATH / "conformers"


@pytest.fixture
def job_manager(tmp_path: Path) -> Generator[JobManager, None, None]:
    """Run the PLAMS jobs of a test in **tmp_path**."""
    with contextlib.redirect_stdout(io.StringIO()):
        init(path=str(tmp_path), folder="init")  # Load the PLAMS defaults; a no-op if done before
    old = config.default_jobmanager
    manager = JobManager(config.jobmanager, str(tmp_path), "plams_workdir")
    config.default_jobmanager = manager
    try:
        yield manager
    finally:
        config.default_jobmanager = old


def test_unique() -> None:
    """Conformers with the same energy and moments of inertia are duplicates."""
    mol = from_smiles("CCCC")
    mol_list = []
    for energy, scale in [(0.0, 1.0), (0.05, 1.0), (0.05, 1.1), (0.5, 1.0)]:
        m = mol.copy()
        m.from_array(mol.as_array() * scale)
        m.properties.energy.E = energy
        mol_list.append(m)
    assert list(_unique(mol_list)) == [mol_list[0], mol_list[2], mol_list[3]]


@requires_tblite
def test_conformer_search(job_manager: JobManager) -> None:
    mol = from_smiles("CCCCCC(=O)[O-]")
    xyz = mol.as_array()
    ret = conformer_search(mol, n=8, parallel_jobs=2)
    np.testing.assert_array_equal(mol.as_array(), xyz)  # The input is not modified

    assert 1 < len(ret) <= 9
    e_rel = [m.properties.energy.E_rel for m in ret]
    assert e_rel[0] == 0
    assert np.all(np.diff(e_rel) >= 0)
    assert len({m.properties.conformer for m in ret}) == len(ret)
    assert all(m.properties.is_opt for m in ret)
    assert len(list(_unique(ret))) == len(ret)


@requires_tblite
def test_conformer_search_all(job_manager: JobManager) -> None:
    """Without removing duplicates, all successful optimizations are returned."""
    mol = from_smiles("CCCCCC(=O)[O-]")
    ret = conformer_search(mol, n=20, unique=False)
    assert len(ret) >= len(list(_unique(ret)))
    assert np.all(np.diff([m.properties.energy.E for m in ret]) >= 0)


@requires_tblite
def test_conformer_search_fail(job_manager: JobManager) -> None:
    mol = from_smiles("CCO")
    assert conformer_search(mol, n=2, settings=Settings({"input": {"foo": 1}})) == []


@requires_tblite
class TestWorkflow:
    SETTINGS = Settings({
        "path": str(WORKDIR),
        "input_cores": [str(PATH / "core" / "Cd68Se55.xyz")],
        "input_ligands": ["CCCC(=O)O", "OC(=O)CCC(C)C"],
        "optional": {
            "core": {"dummy": "Cl"},
            "qd": {"construct_qd": False},
            "parallel_jobs": 2,
            "ligand": {"split": True, "conformers": {"n": 4}},
        },
    })

    @pytest.fixture(scope="function", autouse=True)
    def setup_cat(self) -> Generator[None, None, None]:
        os.makedirs(WORKDIR, exist_ok=True)
        yield None
        shutil.rmtree(WORKDIR, ignore_errors=True)

    def test_pass(self) -> None:
        *_, ligand_df = prep(self.SETTINGS.copy())
        count = ligand_df[CONFORMERS_COUNT]
        e_rel = ligand_df[CONFORMERS_E_REL]
        assert (count >= 1).all()
        assert (e_rel >= 0).all()

        for mol, n in zip(ligand_df["mol"], count):
            filename = WORKDIR / "ligand" / f"{mol.properties.name}.conformers.xyz"
            lines = filename.read_text().splitlines()
            assert len(lines) == n * (len(mol) + 2)

        # The results are read back from the database
        *_, ligand_df2 = prep(self.SETTINGS.copy())
        pd.testing.assert_series_equal(ligand_df2[CONFORMERS_COUNT], count)
        pd.testing.assert_series_equal(ligand_df2[CONFORMERS_E_REL], e_rel)
