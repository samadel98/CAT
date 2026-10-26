"""Tests for :mod:`CAT.data_handling.validate_input`."""

import os
from os.path import join
from pathlib import Path
from itertools import chain

import yaml
import pytest
from unittest import mock

from rdkit import Chem
from scm.plams import (Settings, AMSJob)
from assertionlib import assertion
from nanoutils import delete_finally

from CAT.data_handling.validate_input import validate_input
from CAT.utils import AllignmentTup, AllignmentEnum

try:
    from dataCAT import Database
except Exception:  # Data-CAT 0.7.2 raises an `AttributeError` on NumPy 2
    HAS_DATACAT = False
else:
    HAS_DATACAT = True

PATH = Path('tests') / 'test_files'
LIG_PATH = PATH / 'ligand'
QD_PATH = PATH / 'qd'
DB_PATH = PATH / 'database'


@mock.patch.dict(
    os.environ,
    {'AMSBIN': '', 'AMSHOME': '', 'AMSRESOURCES': '', 'SCMLICENSE': ''},
)
@delete_finally(LIG_PATH, QD_PATH, DB_PATH)
def test_validate_input() -> None:
    """Test :func:`CAT.data_handling.validate_input.validate_input`."""
    with open(join(PATH, 'input1.yaml')) as f:
        s = Settings(yaml.load(f, Loader=yaml.FullLoader))
    s.path = PATH
    validate_input(s, validate_only=False)

    s2 = s.copy()
    validate_input(s2, validate_only=False)

    ref = Settings()
    ref.core.dirname = join(PATH, 'core')
    ref.core.allignment = AllignmentTup(AllignmentEnum.SURFACE, False)
    ref.core.subset = None

    ref.database.dirname = join(PATH, 'database')
    ref.database.mol_format = ('pdb',)
    ref.database.mongodb = {}
    ref.database.overwrite = ()
    ref.database.read = ('core', 'ligand', 'qd')
    ref.database.write = ('core', 'ligand', 'qd')
    ref.database.thread_safe = False
    ref.processes = 1
    ref.parallel_jobs = 1
    if HAS_DATACAT:
        ref.database.db = Database(ref.database.dirname, **ref.database.mongodb)
    else:
        ref.database.db = None

    ref.ligand['cosmo-rs'] = False
    ref.ligand.dirname = join(PATH, 'ligand')
    ref.ligand.optimize = {'job1': None, 'job2': None, 's1': None, 's2': Settings(),
                           'use_ff': False, 'keep_files': True}
    ref.ligand.split = True
    ref.ligand.cdft = False
    ref.ligand.cone_angle = False
    ref.ligand.branch_distance = False

    ref.qd.bulkiness = False
    ref.qd.construct_qd = True
    ref.qd.activation_strain = False
    ref.qd.dirname = join(PATH, 'qd')
    ref.qd.dissociate = False
    ref.qd.multi_ligand = None
    ref.qd.optimize = {
        'job1': AMSJob,
        'keep_files': True,
        'use_ff': False,
        's2': {
            'description': 'UFF with the default forcefield',
            'input': {
                'forcefield': {'type': 'uff'},
                'ams': {'system': {'bondorders': {}}}
            },
        },
        's1': {
            'description': 'UFF with the default forcefield',
            'input': {
                'forcefield': {'type': 'uff'},
                'ams': {'system': {'bondorders': {}}}
            },
        },
        'job2': AMSJob,
    }

    ref.forcefield = Settings()

    lig_func_groups1 = s.optional.ligand.pop('anchor')
    lig_func_groups2 = s2.optional.ligand.pop('anchor')
    for tup in chain(lig_func_groups1, lig_func_groups2):
        assertion.isinstance(tup.mol, Chem.Mol)

    core_func_groups1 = s.optional.core.pop('anchor')
    core_func_groups2 = s2.optional.core.pop('anchor')
    for tup in chain(core_func_groups1, core_func_groups2):
        assertion.isinstance(tup.mol, Chem.Mol)

    assertion.eq(s.optional, ref)
    assertion.eq(s2.optional, ref)


def test_validate_input2() -> None:
    """Test :func:`CAT.data_handling.validate_input.validate_input`."""
    with open(join(PATH, 'input1.yaml')) as f:
        s = Settings(yaml.load(f, Loader=yaml.FullLoader))
    validate_input(s, validate_only=True)

    s2 = s.copy()
    validate_input(s2, validate_only=True)


BDE_PASS = {
    "core_atom": {"core_atom": "Cd", "core_index": None, "lig_count": 2},
    "core_index": {"core_atom": None, "core_index": [1, 2], "lig_count": 2},
}

BDE_RAISE = {
    "neither": ({"core_atom": None, "core_index": None, "lig_count": 2}, "Exactly one of"),
    "both": ({"core_atom": "Cd", "core_index": [1], "lig_count": 2}, "Exactly one of"),
    "lig_count": ({"core_atom": "Cd", "core_index": None, "lig_count": None}, "`lig_count`"),
}


@pytest.mark.parametrize("bde", BDE_PASS.values(), ids=BDE_PASS)
def test_validate_bde(bde: dict) -> None:
    """Test the validation of ``optional.qd.dissociate``."""
    with open(join(PATH, 'input1.yaml')) as f:
        s = Settings(yaml.load(f, Loader=yaml.FullLoader))
    s.optional.qd.dissociate = bde
    validate_input(s, validate_only=True)

    dissociate = s.optional.qd.dissociate
    assertion.eq(dissociate.core_atom, None if bde["core_atom"] is None else 48)  # Cd
    assertion.eq(dissociate.lig_count, bde["lig_count"])
    assertion.eq(dissociate.core_index, None if bde["core_index"] is None else (1, 2))


@pytest.mark.parametrize("bde,match", BDE_RAISE.values(), ids=BDE_RAISE)
def test_validate_bde_raise(bde: dict, match: str) -> None:
    """Test that invalid ``optional.qd.dissociate`` blocks are rejected."""
    with open(join(PATH, 'input1.yaml')) as f:
        s = Settings(yaml.load(f, Loader=yaml.FullLoader))
    s.optional.qd.dissociate = bde
    with pytest.raises(ValueError, match=match):
        validate_input(s, validate_only=True)
