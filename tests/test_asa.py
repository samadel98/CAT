"""Tests for :mod:`nanoCAT.asa.asa`."""

from pathlib import Path

import numpy as np
from scm.plams import Molecule, from_smiles

from nanoCAT.asa.asa import _asa_uff

PATH = Path('tests') / 'test_files'


def test_asa_uff() -> None:
    """Test the RDKit UFF-based activation strain analysis of a core with ligands."""
    core = Molecule(PATH / 'Cd68Se55.xyz')
    for at in core:
        at.properties.charge = 2 if at.symbol == 'Cd' else -2

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
