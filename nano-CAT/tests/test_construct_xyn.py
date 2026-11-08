"""Tests for :mod:`nanoCAT.bde.construct_xyn`."""

from __future__ import annotations

import io
import contextlib
import importlib.util
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from scm.plams import Atom, JobManager, Molecule, config, init

from CAT._compat import from_smiles
from CAT.mol_utils import rdkit_props
from nanoCAT.bde import construct_xyn
from nanoCAT.bde.construct_xyn import get_xyn

requires_tblite = pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="requires tblite"
)


def get_mol_ref(smiles: str) -> Molecule:
    """Return a mock quantum dot: a Cd atom (residue 1) and a ligand (residue 2)."""
    lig = from_smiles(smiles)
    for at in lig:
        rdkit_props(at).pdb_info.ResidueNumber = 2
        rdkit_props(at).pdb_info.ResidueName = "LIG"
    anchor = next(at for at in lig if rdkit_props(at).get("charge", 0) < 0)
    anchor.properties.anchor = True

    ret = Molecule()
    core_atom = Atom(symbol="Cd", coords=(10.0, 10.0, 10.0))
    rdkit_props(core_atom).pdb_info.ResidueNumber = 1
    ret.add_atom(core_atom)
    ret.add_molecule(lig)
    ret.properties.name = "mol_ref"
    ret.properties.path = None
    ret.properties.job_path = []
    return ret


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


@pytest.mark.parametrize("ion,lig_count", [("Cd", 2), ("In", 3), ("Pb", 4)])
def test_monoatomic_ligands(ion: str, lig_count: int) -> None:
    """Monoatomic ligands (*e.g.* halides) must be spread around the ion, in a plane."""
    xyn = get_xyn(get_mol_ref("[Cl-]"), lig_count=lig_count, ion=ion, opt=False)
    xyz = np.array(xyn)
    assert [at.symbol for at in xyn] == [ion] + lig_count * ["Cl"]

    dist = np.linalg.norm(xyz[1:] - xyz[0], axis=1)
    np.testing.assert_allclose(dist, dist[0], atol=1e-3)  # XYn.round_coords()
    pair_dist = np.linalg.norm(xyz[1:, None] - xyz[None, 1:], axis=-1)
    assert (pair_dist[~np.eye(lig_count, dtype=bool)] > dist[0]).all()
    np.testing.assert_allclose(xyz[1:].mean(axis=0), xyz[0], atol=1e-3)
    assert sum(rdkit_props(at).get("charge", 0) for at in xyn) == 0


@requires_tblite
@pytest.mark.parametrize("smiles", ["[Cl-]", "CC(=O)[O-]"])
def test_preoptimize_tblite(
    smiles: str, job_manager: JobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without AMS, :math:`XY_{n}` is pre-optimized with GFN2-xTB, keeping the ion frozen."""
    monkeypatch.delenv("AMSBIN", raising=False)
    mol_ref = get_mol_ref(smiles)
    ref = get_xyn(mol_ref, lig_count=2, ion="Cd", opt=False)
    xyn = get_xyn(mol_ref, lig_count=2, ion="Cd", opt=True)

    xyz_ref = np.array(ref)
    xyz = np.array(xyn)
    np.testing.assert_allclose(xyz[0], xyz_ref[0])
    assert not np.allclose(xyz, xyz_ref, atol=0.01)
    assert len(xyn.properties.job_path) == 1
    assert xyn.properties.job_path[0].endswith("E_XYn_preopt.in")


def test_preoptimize_skip(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without AMS and tblite, the pre-optimization is skipped with a warning."""
    monkeypatch.delenv("AMSBIN", raising=False)
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        construct_xyn.importlib.util, "find_spec",
        lambda name, *args: None if name == "tblite" else find_spec(name, *args),
    )
    mol_ref = get_mol_ref("CC(=O)[O-]")
    ref = get_xyn(mol_ref, lig_count=2, ion="Cd", opt=False)
    with caplog.at_level("WARNING", logger="CAT"):
        xyn = get_xyn(mol_ref, lig_count=2, ion="Cd", opt=True)

    np.testing.assert_allclose(np.array(xyn), np.array(ref))
    assert "skipping the pre-optimization" in caplog.text
