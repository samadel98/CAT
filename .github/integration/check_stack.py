"""Run CAT end to end with Nano-CAT, Data-CAT and Auto-FOX, as a user would.

Usage: ``python check_stack.py <workdir> [--mongodb]``

The input below builds quantum dots from one core and three ligands, runs the Nano-CAT
workflows that need no external programs, and writes everything to the database.
The script then checks the results, runs the same input again (which must read every result
back from the database instead of recomputing it) and, with ``--mongodb``, exports the
results to a MongoDB server on localhost and reads them back.

"""

from __future__ import annotations

import sys
import shutil
import logging
import argparse
import subprocess
from pathlib import Path

import yaml
import pymongo
import numpy as np
import pandas as pd
from scm.plams import Settings

import CAT
import FOX
import nanoCAT
import dataCAT
from CAT.base import prep
from CAT.workflows import MOL

CORE = Path(CAT.__file__).parents[1] / 'tests' / 'test_files' / 'core' / 'Cd68Se55.xyz'

INPUT = """
path: {path}

input_cores:
    - Cd68Se55.xyz:
        guess_bonds: False

input_ligands:
    - OC(C)=O
    - OC(=O)CC(C)C
    - OC(=O)c1ccccc1

optional:
    processes: 2
    database:
        read: True
        write: True
        mol_format: [pdb, xyz]
    core:
        dummy: Cl
    ligand:
        optimize: True
        split: True
        cone_angle: True
        branch_distance: True
    qd:
        construct_qd: True
        activation_strain: True
        bulkiness: True
"""

#: Float columns that the workflows above must fill
LIGAND_COLUMNS = [('cone_angle', 'dist=0.0')]
QD_COLUMNS = [('ASA', 'E_int'), ('ASA', 'E_strain'), ('ASA', 'E'), ('V_bulk', '')]


class LogRecorder(logging.Handler):
    """Store the messages of all log records."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the message of **record**."""
        self.messages.append(record.getMessage())


def run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Run CAT on the input; return the ligand and QD results and the log messages."""
    s = Settings(yaml.safe_load((path / 'input.yaml').read_text()))
    recorder = LogRecorder()
    logger = logging.getLogger('CAT')
    logger.addHandler(recorder)
    try:
        qd_df, _, ligand_df = prep(s)
    finally:
        logger.removeHandler(recorder)
    return ligand_df, qd_df, recorder.messages


def check_columns(df: pd.DataFrame, columns: list[tuple[str, str]], name: str) -> None:
    """Check that **df** has finite values in all **columns**."""
    for col in columns:
        assert col in df.columns, f'{name}: missing column {col}; got {df.columns.tolist()}'
        values = df[col].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f'{name}: {col} is not finite: {values}'


def main(workdir: Path, mongodb: bool) -> None:
    """Run all checks in **workdir**."""
    for module in (CAT, nanoCAT, dataCAT, FOX):
        print(f'{module.__name__} {module.__version__}: {Path(module.__file__).parent}')

    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / 'core').mkdir(parents=True)
    shutil.copy(CORE, workdir / 'core')
    (workdir / 'input.yaml').write_text(INPUT.format(path=workdir.as_posix()))

    # The dry run must find Nano-CAT and Data-CAT
    out = subprocess.run(
        ['init_cat', '--dry-run', str(workdir / 'input.yaml')],
        check=True, capture_output=True, text=True,
    ).stdout
    print(out)
    assert 'could not be imported' not in out, out
    assert 'Read 1 core(s) and 3 ligand(s)' in out, out

    # First run: compute everything
    ligand_df, qd_df, _ = run(workdir)
    assert len(ligand_df) == 3, ligand_df.index
    assert len(qd_df) == 3, qd_df.index
    assert ligand_df['opt'].all() and ligand_df['hdf5 index'].ge(0).all()
    check_columns(ligand_df, LIGAND_COLUMNS, 'ligands')
    check_columns(qd_df, QD_COLUMNS, 'quantum dots')
    assert sorted(len(mol) for mol in qd_df[MOL]) == [123 + 26 * n for n in (7, 14, 16)]
    branches = ligand_df[[('branch_distance', ''), ('branch_size', '')]].astype(str)
    assert branches.values.tolist() == [['0', '7'], ['0', '16'], ['0-1', '11-3']], branches
    for name in ('ligand', 'qd'):
        for ext in ('pdb', 'xyz'):
            assert len(list((workdir / name).glob(f'*.{ext}'))) == 3, (name, ext)

    # Second run: everything except the (cheap) quantum dot construction must come from
    # the database; workflows log "Finishing ..." only if they computed something
    ligand_df2, qd_df2, messages = run(workdir)
    recomputed = [
        msg for msg in messages
        if msg.startswith('UFFGetMoleculeForceField') or (
            msg.startswith('Finishing ') and 'quantum dot construction' not in msg
        )
    ]
    assert not recomputed, f'The second run recomputed results: {recomputed}'
    pd.testing.assert_frame_equal(
        ligand_df2.drop(columns=MOL), ligand_df.drop(columns=MOL), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        qd_df2.drop(columns=MOL), qd_df.drop(columns=MOL), check_dtype=False
    )
    # The ligands are stored in single precision
    for mol1, mol2 in zip(ligand_df[MOL], ligand_df2[MOL]):
        np.testing.assert_allclose(np.array(mol1), np.array(mol2), atol=1e-4)
    for mol1, mol2 in zip(qd_df[MOL], qd_df2[MOL]):
        np.testing.assert_allclose(np.array(mol1), np.array(mol2), atol=1e-4)
        assert max(bond.length() for bond in mol2.bonds) < 2, mol2.properties.name

    if mongodb:
        db = dataCAT.Database(workdir / 'database', host='localhost', port=27017)
        assert db.mongodb is not None, 'No MongoDB server found on localhost'
        db.update_mongodb('ligand', overwrite=True)
        db.update_mongodb('qd', overwrite=True)
        with pymongo.MongoClient(**db.mongodb) as client:
            ligand_docs = list(client.cat_database.ligand_database.find(
                {'smiles': {'$in': ligand_df.index.get_level_values('smiles').tolist()}}
            ))
            qd_docs = list(client.cat_database.qd_database.find(
                {'ligand smiles': {'$in': qd_df.index.get_level_values('ligand smiles').tolist()}}
            ))
        assert (len(ligand_docs), len(qd_docs)) == (3, 3), (ligand_docs, qd_docs)
        cone_angles = sorted(doc['cone_angle']['dist=0.0'] for doc in ligand_docs)
        np.testing.assert_allclose(cone_angles, sorted(ligand_df[LIGAND_COLUMNS[0]]))
        print(f'MongoDB: found the {len(ligand_docs)} ligands and {len(qd_docs)} quantum dots')

    print('All checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--mongodb', action='store_true', help='Also check the MongoDB export')
    args = parser.parse_args()
    main(args.workdir.resolve(), args.mongodb)
    sys.exit(0)
