"""Tests for :mod:`CAT.parallel` and the ``optional.processes`` option."""

from __future__ import annotations

import os
import pickle
import logging
import operator
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from schema import SchemaError
from scm.plams import Molecule, Settings, from_smiles

from CAT.base import prep
from CAT.utils import FormatEnum
from CAT.workflows import MOL
from CAT import parallel
from CAT.parallel import get_processes, parallel_map, thread_map, pack_atom_refs, unpack_atom_refs
from CAT.attachment.ligand_attach import _copy_mol
from CAT.mol_utils import rdkit_props
from CAT.data_handling.validation_schemas import processes_schema, parallel_jobs_schema

PATH = Path('tests') / 'test_files'

LIGANDS = ['[O-]C(=O)C', '[O-]C(=O)CC(C)C', '[O-]C(=O)c1ccccc1', '[O-]C(=O)CCCCCCC/C=C\\CCCCCCCC']


@pytest.mark.parametrize("value,ref", [(1, 1), (3, 3), (None, os.cpu_count() or 1),
                                       (0, os.cpu_count() or 1)])
def test_get_processes(value: None | int, ref: int) -> None:
    """Test :func:`CAT.parallel.get_processes`."""
    assert get_processes(processes_schema.validate(value)) == ref


@pytest.mark.parametrize("schema", [processes_schema, parallel_jobs_schema])
@pytest.mark.parametrize("value", [-1, True, 1.5, "4"])
def test_processes_schema_raise(value: Any, schema: Any) -> None:
    """Test that :data:`processes_schema` and :data:`parallel_jobs_schema` reject invalid values."""
    with pytest.raises(SchemaError):
        schema.validate(value)


def test_parallel_map() -> None:
    """Test :func:`CAT.parallel.parallel_map`."""
    args = [(i, 2 * i) for i in range(10)]
    ref = [3 * i for i in range(10)]
    assert parallel_map(operator.add, args, 1) == ref
    assert parallel_map(operator.add, args, 2) == ref
    assert parallel_map(operator.add, [], 2) == []


def _get_omp_threads(i: int) -> tuple[int, str | None]:
    return i, os.environ.get("OMP_NUM_THREADS")


def _raise_for_odd(i: int) -> int:
    if i % 2:
        raise ValueError(i)
    return i


def test_thread_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test :func:`CAT.parallel.thread_map`."""
    args = [(i, 2 * i) for i in range(10)]
    ref = [3 * i for i in range(10)]
    assert thread_map(operator.add, args, 1) == ref
    assert thread_map(operator.add, args, 4) == ref
    assert thread_map(operator.add, [], 4) == []

    # The threads' child processes get a single OpenMP thread, unless set otherwise
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert thread_map(_get_omp_threads, [(i,) for i in range(4)], 2) == [(i, "1") for i in range(4)]
    assert "OMP_NUM_THREADS" not in os.environ
    assert thread_map(_get_omp_threads, [(0,)], 2) == [(0, None)]

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert thread_map(_get_omp_threads, [(0,), (1,)], 2) == [(0, "3"), (1, "3")]

    # Exceptions are re-raised
    with pytest.raises(ValueError, match="^1$"):
        thread_map(_raise_for_odd, [(i,) for i in range(4)], 2)


def test_parallel_map_broken(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the worker processes fail, :func:`parallel_map` falls back to the current process."""
    class BrokenExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> BrokenExecutor:
            raise BrokenProcessPool("A process in the process pool was terminated abruptly")

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", BrokenExecutor)
    with caplog.at_level(logging.WARNING, logger="CAT"):
        assert parallel_map(operator.add, [(1, 2), (3, 4)], 2) == [3, 7]
    assert "if __name__ == '__main__':" in caplog.text


def test_atom_refs() -> None:
    """Test :func:`CAT.parallel.pack_atom_refs` and :func:`CAT.parallel.unpack_atom_refs`."""
    mol = from_smiles('CCO')
    mol.properties.dummies = mol[3]
    mol.properties.atom_list = mol.atoms[:2]
    mol.properties.other_mol_atom = from_smiles('C')[1]
    refs = pack_atom_refs(mol)
    assert refs == {'dummies': 2, 'atom_list': [0, 1]}

    mol2 = pickle.loads(pickle.dumps(mol))
    assert mol2.properties.dummies is not mol2[3]
    unpack_atom_refs(mol2, refs)
    assert mol2.properties.dummies is mol2[3]
    assert mol2.properties.atom_list == mol2.atoms[:2]


@pytest.mark.parametrize("member", list(FormatEnum), ids=lambda i: i.name)
def test_pickle_format_enum(member: FormatEnum) -> None:
    """Test that :class:`CAT.utils.FormatEnum` members can be sent to worker processes."""
    assert pickle.loads(pickle.dumps(member)) is member


def test_copy_mol() -> None:
    """Test that :func:`CAT.attachment.ligand_attach._copy_mol` matches :meth:`Molecule.copy`."""
    mol = Molecule(str(PATH / 'core' / 'Cd68Cl26Se55__26_CC[O-]@O3.pdb'))
    mol.properties.name = 'test'
    for i, at in enumerate(mol, 1):
        rdkit_props(at).pdb_info = Settings({'Name': f'{at.symbol}{i}', 'ResidueNumber': 1})
    ref, copy = mol.copy(), _copy_mol(mol)
    assert copy.as_dict() == ref.as_dict()
    assert copy.properties == ref.properties
    assert all(at.mol is copy for at in copy) and all(b.mol is copy for b in copy.bonds)
    assert all(b.atom1 in copy and b.atom2 in copy for b in copy.bonds)

    rdkit_props(copy[1]).pdb_info.ResidueNumber = 100
    assert rdkit_props(mol[1]).pdb_info.ResidueNumber == 1


def _state(mol: Molecule) -> dict[str, Any]:
    """Return everything about **mol** that an optimization can change."""
    atom_idx = {id(at): i for i, at in enumerate(mol.atoms)}
    bond_idx = {id(b): i for i, b in enumerate(mol.bonds)}
    return {
        'atoms': [(at.symbol, at.coords, at.properties.as_dict()) for at in mol],
        'bonds': [(atom_idx[id(b.atom1)], atom_idx[id(b.atom2)], b.order) for b in mol.bonds],
        'atom_bonds': [[bond_idx[id(b)] for b in at.bonds] for at in mol],
        'dummies': atom_idx[id(mol.properties.dummies)],
    }


def test_processes(tmp_path: Path) -> None:
    """Test that the ligands and quantum dots do not depend on ``optional.processes``."""
    ret = {}
    for processes in (1, 2):
        s = Settings({
            'path': str(tmp_path),
            'input_cores': [{str(PATH / 'core' / 'Cd68Se55.xyz'): {'guess_bonds': False}}],
            'input_ligands': LIGANDS,
            'optional': {
                'processes': processes,
                'core': {'dummy': 'Cl', 'dirname': f'core_{processes}'},
                'ligand': {'optimize': True, 'split': False, 'dirname': f'ligand_{processes}'},
                'qd': {'dirname': f'qd_{processes}'},
                'database': {'read': False, 'write': False, 'dirname': f'database_{processes}'},
            },
        })
        ret[processes] = prep(s)

    (qd_df1, _, lig_df1), (qd_df2, _, lig_df2) = ret[1], ret[2]
    assert lig_df1.index.equals(lig_df2.index) and len(lig_df1) == len(LIGANDS)
    for lig1, lig2 in zip(lig_df1[MOL], lig_df2[MOL]):
        assert lig1.properties.is_opt and lig2.properties.is_opt
        assert _state(lig1) == _state(lig2)
        np.testing.assert_array_equal(np.array(lig2), [at.coords for at in lig2])
    for qd1, qd2 in zip(qd_df1[MOL], qd_df2[MOL]):
        np.testing.assert_array_equal(np.array(qd1), np.array(qd2))
