"""Tests for :mod:`CAT.ase`."""

from __future__ import annotations

import io
import json
import contextlib
import importlib.util
from pathlib import Path
from typing import Any
from collections.abc import Generator

import numpy as np
import pytest
from scm.plams import JobManager, ResultsError, Settings, config, from_smiles, init

from CAT.jobs import get_job_template, freeze_atoms
from CAT.ase import ASEJob, check_ase
from CAT.xtb import TBLiteJob, runner as xtb_runner
from CAT.data_handling.validate_input import tblite_settings
from CAT.data_handling.validation_schemas import qd_opt_schema, str_to_job_type

pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.units import Hartree  # noqa: E402
from ase.calculators.lj import LennardJones  # noqa: E402
from CAT.ase import runner  # noqa: E402

requires_tblite = pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="requires tblite"
)

#: Water, with the atomic symbol, atomic number, mass and coordinates of each atom
WATER = [
    ["O", 8, 15.999, 0.0, 0.0, 0.1],
    ["H", 1, 1.008, 0.0, 0.8, -0.5],
    ["H", 1, 1.008, 0.0, -0.8, -0.5],
]


def _quiet(msg: str) -> None:
    pass


class TestGetCalculator:
    def test_alias(self) -> None:
        calc = runner.get_calculator("LJ", {"sigma": 2.0})
        assert isinstance(calc, LennardJones)
        assert calc.parameters.sigma == 2.0

    @pytest.mark.parametrize("name", ["ase.calculators.lj.LennardJones",
                                      "ase.calculators.lj:LennardJones"])
    def test_import_path(self, name: str) -> None:
        assert isinstance(runner.get_calculator(name), LennardJones)

    @pytest.mark.parametrize("name,exc", [
        (None, TypeError),
        ("foo", ValueError),
        ("ase.calculators.lj.Foo", AttributeError),
    ], ids=["none", "alias", "attribute"])
    def test_raise(self, name: Any, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            runner.get_calculator(name)


def test_run_lj() -> None:
    """Compute the energy and gradient of water with a Lennard-Jones potential."""
    inp = {"calculator": "lj", "calculator_kwargs": {"sigma": 1.5}}
    ret = runner.run({"input": inp, "atoms": WATER}, log=_quiet)

    atoms = Atoms("OH2", positions=[at[3:] for at in WATER])
    atoms.calc = LennardJones(sigma=1.5)
    assert ret["energy"] == pytest.approx(atoms.get_potential_energy() / Hartree, rel=1e-12)
    np.testing.assert_allclose(
        ret["gradient"], -atoms.get_forces() / Hartree * xtb_runner.BOHR, rtol=1e-10
    )
    assert ret["charges"] is None
    assert ret["converged"]
    assert ret["method"] == "lj"

    # A geometry optimization with a frozen atom
    inp = {"task": "geometry", "calculator": "lj", "constraints": {"atom": [1]}}
    ret2 = runner.run({"input": inp, "atoms": WATER}, log=_quiet)
    assert ret2["converged"]
    assert ret2["energy"] < ret["energy"]
    np.testing.assert_allclose(ret2["positions"][0], WATER[0][3:], atol=1e-12)

    with pytest.raises(ValueError):
        runner.run({"input": {"calculator": "lj", "task": "solvation"}, "atoms": WATER})


@requires_tblite
def test_run_tblite() -> None:
    """The tblite ASE calculator gives the same results as :mod:`CAT.xtb.runner`."""
    ret1 = xtb_runner.run({"input": {"task": "frequencies"}, "atoms": WATER}, log=_quiet)
    inp = {"task": "frequencies", "calculator": "tblite", "calculator_kwargs": {"verbosity": 0}}
    ret2 = runner.run({"input": inp, "atoms": WATER}, log=_quiet)
    assert ret2["energy"] == pytest.approx(ret1["energy"], abs=1e-6)
    np.testing.assert_allclose(ret2["gradient"], ret1["gradient"], atol=1e-6)
    np.testing.assert_allclose(ret2["charges"], ret1["charges"], atol=1e-6)
    np.testing.assert_allclose(ret2["frequencies"], ret1["frequencies"], atol=1)


def test_run_main(tmp_path: Path) -> None:
    """Test the files written by :func:`CAT.ase.runner.main`."""
    inp_file = tmp_path / "water.in"
    inp_file.write_text(json.dumps({"input": {"calculator": "lj"}, "atoms": WATER}))
    assert runner.main([str(inp_file)]) == 0
    ret = json.loads((tmp_path / "water.json").read_text())
    assert ret["task"] == "singlepoint"
    assert (tmp_path / "water.xyz").is_file()

    inp_file.write_text(json.dumps({"input": {}, "atoms": WATER}))
    assert runner.main([str(inp_file)]) == 1


def test_check_ase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test :func:`CAT.ase.check_ase`."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError):
        check_ase()


def test_cat_integration() -> None:
    """Test the job presets, constraints and input validation of an :class:`ASEJob`."""
    assert get_job_template(ASEJob, "geometry") == {"task": "geometry"}
    assert freeze_atoms(ASEJob, Settings(), [1, 2]) == {"input": {"constraints": {"atom": [1, 2]}}}
    assert str_to_job_type("ase") is ASEJob
    raw = {"job1": "ase", "job2": None}
    ret = tblite_settings(raw, qd_opt_schema.validate(raw))
    assert ret["job1"] is ASEJob
    assert ret["s1"] == Settings()


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


def test_ase_job(job_manager: JobManager) -> None:
    """Run an :class:`ASEJob` with the job presets of :mod:`CAT.jobs`."""
    mol = from_smiles("CC(=O)[O-]")
    s = Settings({"input": {"calculator": "lj", "calculator_kwargs": {"Sigma": 1.5}}})
    mol.job_single_point(ASEJob, s)
    assert np.isfinite(mol.properties.energy.E)

    # The keys of `calculator_kwargs` keep their case; the charge is the sum of the formal charges
    inp = json.loads(Path(mol.properties.job_path[0]).read_text())["input"]
    assert inp["calculator_kwargs"] == {"Sigma": 1.5}
    assert inp["charge"] == -1
    assert sorted(inp["formal_charges"]) == [-1] + [0] * (len(mol) - 1)

    results = ASEJob(molecule=mol, settings=Settings({"input": {"calculator": "lj"}})).run()
    with pytest.raises(ResultsError):
        results.get_charges()

    # Errors make the job fail
    mol2 = from_smiles("CCO")
    mol2.job_single_point(ASEJob, Settings({"input": {"calculator": "foo"}}))
    assert np.isnan(mol2.properties.energy.E)


@requires_tblite
def test_ase_job_tblite(job_manager: JobManager) -> None:
    """An :class:`ASEJob` with tblite optimizes like a :class:`TBLiteJob`, including the charge."""
    mol1 = from_smiles("CC(=O)[O-]")
    mol2 = mol1.copy()
    mol1.job_geometry_opt(TBLiteJob, Settings())
    s = Settings({"input": {"calculator": "tblite", "calculator_kwargs": {"verbosity": 0}}})
    mol2.job_geometry_opt(ASEJob, s)
    assert mol2.properties.is_opt
    assert mol2.properties.energy.E == pytest.approx(mol1.properties.energy.E, abs=1e-4)
    np.testing.assert_allclose(mol2.as_array(), mol1.as_array(), atol=1e-3)
