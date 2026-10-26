"""Tests for :mod:`nanoCAT.asa.asa`."""

from pathlib import Path
from typing import Any

import pytest
import numpy as np
from scm.plams import Atom, Molecule
from CAT._compat import from_smiles
from CAT.mol_utils import rdkit_props

from nanoCAT.asa.asa import _asa_uff, _asa_plams

PATH = Path('tests') / 'test_files'


def test_asa_uff() -> None:
    """Test the RDKit UFF-based activation strain analysis of a core with ligands."""
    core = Molecule(PATH / 'Cd68Se55.xyz')
    for at in core:
        rdkit_props(at).charge = 2 if at.symbol == 'Cd' else -2

    ligands = []
    for i in range(3):
        lig = from_smiles('CCC(=O)[O-]')
        lig.translate(np.array([0.0, 0.0, 20.0 + 10.0 * i]))
        ligands.append(lig)

    complete = core.copy()
    for lig in ligands:
        complete += lig.copy()

    # The core's RDKit molecule used to be freed before its UFF energy was computed,
    # which crashed the interpreter (RDKit >= 2025)
    E_complete, E_ligands, E_core, E_ligand_opt, count = _asa_uff(
        complete, ligands, core, read_template=False, job=None, settings=None,
    )
    assert count == 3
    assert np.isfinite([E_complete, E_ligands, E_core, E_ligand_opt]).all()
    assert E_ligand_opt <= E_ligands / 3 + 1e-8


def _fake_job(self: Molecule, *args: Any, **kwargs: Any) -> None:
    """Set the energy of a molecule to its number of atoms."""
    self.properties.energy.E = float(len(self))


def test_asa_plams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the PLAMS-based activation strain analysis reports the energy of the core."""
    monkeypatch.setattr(Molecule, "job_single_point", _fake_job, raising=False)
    monkeypatch.setattr(Molecule, "job_geometry_opt", _fake_job, raising=False)

    core = Molecule()
    core.add_atom(Atom(symbol="Cd", coords=(0.0, 0.0, 0.0)))
    ligands = [from_smiles("CC(=O)[O-]"), from_smiles("CC(=O)[O-]")]
    complete = core.copy()
    for lig in ligands:
        complete += lig.copy()
    complete.properties.name = "QD"

    E_complete, E_ligands, E_core, E_ligand_opt, count = _asa_plams(
        complete, ligands, core, read_template=False, job=None, settings=None,
    )
    assert count == 2
    assert E_complete == len(complete)
    assert E_ligands == len(ligands[0]) + len(ligands[1])
    assert E_core == len(core)
    assert E_ligand_opt == len(ligands[0])
