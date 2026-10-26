"""Tests for :mod:`CAT.xtb`."""

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
from scm.plams import (
    AMSJob, Cp2kJob, JobManager, Molecule, Settings, config, from_smiles, init
)

from CAT.jobs import get_job_template
from CAT.xtb import TBLiteJob, check_tblite, runner
from CAT.frozen_settings import FrozenSettings
from CAT.jobs import freeze_atoms
from CAT.data_handling.validate_input import tblite_settings
from CAT.data_handling.validation_schemas import qd_opt_schema, bde_schema, str_to_job_type

requires_tblite = pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="requires tblite"
)

#: Water, with the atomic symbol, atomic number, mass and coordinates of each atom
WATER = [
    ["O", 8, 15.999, 0.0, 0.0, 0.1],
    ["H", 1, 1.008, 0.0, 0.8, -0.5],
    ["H", 1, 1.008, 0.0, -0.8, -0.5],
]


class TestParseInput:
    """Tests for :func:`CAT.xtb.runner.parse_input`."""

    def test_defaults(self) -> None:
        assert runner.parse_input({}) == runner.DEFAULTS

    def test_keys(self) -> None:
        """Keys are case insensitive and dashes are equivalent to underscores."""
        inp = runner.parse_input({"Task": "geometry", "Geometry": {"Gradient-Tol": 1e-4}})
        assert inp["task"] == "geometry"
        assert inp["geometry"]["gradient_tol"] == 1e-4
        assert inp["geometry"]["max_iter"] == runner.DEFAULTS["geometry"]["max_iter"]

    @pytest.mark.parametrize("inp,exc", [
        ({"foo": 1}, ValueError),
        ({"geometry": {"foo": 1}}, ValueError),
        ({"geometry": 1}, TypeError),
        ({"task": "md"}, ValueError),
        ({"solvation_model": "cosmo"}, ValueError),
    ], ids=["key", "nested key", "nested type", "task", "solvation model"])
    def test_raise(self, inp: dict[str, Any], exc: type[Exception]) -> None:
        with pytest.raises(exc):
            runner.parse_input(inp)


def test_get_free() -> None:
    """Test :func:`CAT.xtb.runner._get_free`."""
    np.testing.assert_array_equal(runner._get_free([1, 3], 4), [False, True, False, True])
    np.testing.assert_array_equal(runner._get_free(2, 2), [True, False])
    np.testing.assert_array_equal(runner._get_free([], 2), [True, True])
    with pytest.raises(IndexError):
        runner._get_free([0], 2)


@pytest.mark.parametrize("positions,count", [
    ([[0.0, 0.0, 0.0], [0.0, 1.4, 1.1], [0.0, -1.4, 1.1]], 3),
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, -2.2]], 4),
], ids=["bent", "linear"])
def test_vibrational_basis(positions: list[list[float]], count: int) -> None:
    """Test that the basis is orthonormal and orthogonal to translations."""
    masses = np.array([12.0, 16.0, 16.0])
    basis = runner.vibrational_basis(np.array(positions), masses)
    assert basis.shape == (9, count)
    np.testing.assert_allclose(basis.T @ basis, np.eye(count), atol=1e-12)

    translation = np.zeros((3, 3))
    translation[:, 0] = np.sqrt(masses)
    np.testing.assert_allclose(translation.ravel() @ basis, 0.0, atol=1e-12)


def test_frequencies_diatomic() -> None:
    """Test the frequency of a diatomic molecule with a harmonic bond."""
    k = 0.37  # Hartree/Bohr**2
    masses = np.array([1.008, 1.008])
    hess = np.zeros((6, 6))
    hess[2::3, 2::3] = k * np.array([[1, -1], [-1, 1]])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])

    freqs = runner.frequencies(hess, masses, positions)
    mu = masses.prod() / masses.sum()
    np.testing.assert_allclose(freqs, [np.sqrt(k / mu) * runner.AU_TO_CM])
    assert 4000 < freqs[0] < 5000

    # Without projection all 3n eigenvalues are returned
    assert len(runner.frequencies(hess, masses)) == 6


class _Harmonic:
    """A fake :class:`CAT.xtb.runner.Engine` with an anisotropic harmonic potential."""

    def __init__(self, minimum: np.ndarray) -> None:
        self.minimum = minimum
        self.k = np.linspace(0.1, 2.0, minimum.size).reshape(minimum.shape)
        self.count = 0

    def __call__(self, positions: np.ndarray, keep: bool = True) -> tuple[float, np.ndarray]:
        self.count += 1
        dx = positions - self.minimum
        return 0.5 * float((self.k * dx**2).sum()), self.k * dx


def test_optimize() -> None:
    """Test the L-BFGS optimizer with frozen atoms."""
    rng = np.random.default_rng(0)
    minimum = rng.random((5, 3))
    x0 = minimum + rng.normal(scale=0.5, size=(5, 3))
    free = np.array([True, True, False, True, True])

    x, energy, gradient, converged, steps = runner.optimize(
        _Harmonic(minimum), x0, free, gradient_tol=1e-6, log=lambda msg: None
    )
    assert converged
    assert 0 < steps < 100
    np.testing.assert_allclose(x[free], minimum[free], atol=1e-5)
    np.testing.assert_array_equal(x[~free], x0[~free])

    # Every step is limited to `max_step` Angstrom
    x, *_, converged, steps = runner.optimize(
        _Harmonic(minimum), x0, free, max_iter=1, max_step=0.01, log=lambda msg: None
    )
    assert not converged
    assert steps == 1
    np.testing.assert_array_less(
        np.linalg.norm(x - x0, axis=1), 0.01 / runner.BOHR + 1e-12
    )


@requires_tblite
def test_run_water() -> None:
    """Optimize water with GFN2-xTB and compute its frequencies."""
    ret = runner.run({"input": {"task": "geometry"}, "atoms": WATER}, log=lambda msg: None)
    assert ret["converged"]
    assert ret["max_gradient"] <= 1e-3
    assert ret["energy"] == pytest.approx(-5.0705, abs=1e-3)
    np.testing.assert_allclose(np.sum(ret["charges"]), 0.0, atol=1e-6)

    atoms = [at[:3] + pos for at, pos in zip(WATER, ret["positions"])]
    ret2 = runner.run({"input": {"task": "frequencies"}, "atoms": atoms}, log=lambda msg: None)
    assert ret2["energy"] == pytest.approx(ret["energy"], abs=1e-8)
    np.testing.assert_allclose(ret2["frequencies"], [1539, 3646, 3656], rtol=0.01)

    # A frozen atom: the partial Hessian of the two free atoms has 6 eigenvalues
    inp = {"task": "frequencies", "constraints": {"atom": [1]}}
    ret3 = runner.run({"input": inp, "atoms": atoms}, log=lambda msg: None)
    assert len(ret3["frequencies"]) == 6


@requires_tblite
def test_run_main(tmp_path: Path) -> None:
    """Test the files written by :func:`CAT.xtb.runner.main`."""
    inp_file = tmp_path / "water.in"
    inp_file.write_text(json.dumps({"input": {"solvent": "water"}, "atoms": WATER}))
    assert runner.main([str(inp_file)]) == 0

    ret = json.loads((tmp_path / "water.json").read_text())
    assert ret["task"] == "singlepoint"
    assert ret["frequencies"] is None
    mol = Molecule(str(tmp_path / "water.xyz"))
    np.testing.assert_allclose(mol.as_array(), [at[3:] for at in WATER])

    # Errors are reported on the first line of stderr
    inp_file.write_text(json.dumps({"input": {"foo": 1}, "atoms": WATER}))
    assert runner.main([str(inp_file)]) == 1


def test_check_tblite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test :func:`CAT.xtb.check_tblite`."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError):
        check_tblite()


def test_templates() -> None:
    """Test the job presets of a :class:`TBLiteJob`."""
    assert get_job_template(TBLiteJob, "singlepoint") == {"task": "singlepoint"}
    assert get_job_template(TBLiteJob, "geometry") == {"task": "geometry"}
    assert get_job_template(TBLiteJob, "freq") == {"task": "frequencies"}


def test_freeze_atoms() -> None:
    """Test :func:`CAT.jobs.freeze_atoms`."""
    s = FrozenSettings({"input": {"solvent": "water"}})
    s2 = freeze_atoms(TBLiteJob, s, [1, 2])
    assert s2 == {"input": {"solvent": "water", "constraints": {"atom": [1, 2]}}}
    assert "constraints" not in s.input

    s3 = freeze_atoms(AMSJob, s, (3,))
    assert s3 == {"input": {"solvent": "water", "ams": {"constraints": {"atom": [3]}}}}
    assert freeze_atoms(TBLiteJob, None, [1]) == {"input": {"constraints": {"atom": [1]}}}
    assert freeze_atoms(None, None, [1, 2]) is None
    assert freeze_atoms(Cp2kJob, s, [1, 2]) is s


@requires_tblite
def test_validate() -> None:
    """Test that a TBLiteJob without settings does not get the default AMS settings."""
    assert str_to_job_type("xtb") is TBLiteJob
    assert str_to_job_type("TBLite") is TBLiteJob

    raw = {"job1": "tblite", "job2": None}
    ret = tblite_settings(raw, qd_opt_schema.validate(raw))
    assert ret["job1"] is TBLiteJob
    assert ret["s1"] == Settings()

    raw = {"job1": "tblite", "s1": {"input": {"solvent": "water"}}, "job2": "tblite"}
    ret = tblite_settings(raw, bde_schema.validate(raw))
    assert ret["s1"] == {"input": {"solvent": "water"}}
    assert ret["s2"] == Settings()


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


@requires_tblite
def test_tblite_job(job_manager: JobManager) -> None:
    """Run a :class:`TBLiteJob` with the job presets of :mod:`CAT.jobs`."""
    mol = from_smiles("CC(=O)[O-]")
    mol.properties.name = "acetate"
    s = Settings({"input": {"solvent": "water"}})

    mol.job_freq(TBLiteJob, s)
    assert mol.properties.is_opt
    assert len(mol.properties.frequencies) == 3 * len(mol) - 6
    energy = mol.properties.energy
    assert np.isfinite([energy.E, energy.H, energy.S, energy.G]).all()
    assert Path(mol.properties.job_path[0]).parent.parent == Path(job_manager.workdir)

    # The charge is the sum of the formal charges
    inp = json.loads(Path(mol.properties.job_path[0]).read_text())
    assert inp["input"] == {"solvent": "water", "task": "geometry", "charge": -1}

    # Unknown settings and unconverged optimizations make the job fail
    mol2 = from_smiles("CCO")
    mol2.job_geometry_opt(TBLiteJob, Settings({"input": {"geometry": {"max_iter": 1}}}))
    assert mol2.properties.is_opt is False
    assert np.isnan(mol2.properties.energy.E)

    mol2.job_single_point(TBLiteJob, Settings({"input": {"foo": 1}}))
    assert np.isnan(mol2.properties.energy.E)
