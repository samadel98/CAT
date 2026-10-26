"""Tests for :mod:`dataCAT.locking`: several processes sharing a database."""

from __future__ import annotations

import shutil
import threading
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from CAT.workflows import MOL, HDF5_INDEX, OPT, V_BULK

from dataCAT import Database
from dataCAT.locking import LOCK_NAME, get_lock, open_hdf5
from dataCAT.testing_utils import _readpdb as readpdb

PATH = Path('tests') / 'test_files'
DB_PATH = PATH / 'database'
MOL_PATH = (PATH / 'CC[=O][O-]@O4.pdb').resolve()

N_WORKERS = 4
N_UPDATES = 5


def _add_ligands(db_path: str, worker: int) -> None:
    """Add :data:`N_UPDATES` ligands to the database, one at a time."""
    mol = readpdb(str(MOL_PATH))
    db = Database(db_path)
    for i in range(N_UPDATES):
        index = pd.MultiIndex.from_tuples(
            [(f'worker{worker}_{i}', 'O4')], names=['smiles', 'anchor']
        )
        df = pd.DataFrame({MOL: [mol]}, index=index)
        df[HDF5_INDEX] = -1
        df[OPT] = False
        df[V_BULK] = float(100 * worker + i)
        db.update_csv(df, database='ligand', columns=[MOL, V_BULK])


def test_processes(tmp_path: Path) -> None:
    """Processes that update the same database at the same time do not lose each other's results."""
    db_path = tmp_path / 'database'
    shutil.copytree(DB_PATH, db_path)

    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(N_WORKERS, mp_context=ctx) as executor:
        futures = [executor.submit(_add_ligands, str(db_path), i) for i in range(N_WORKERS)]
        for future in futures:
            future.result()

    db = Database(db_path)
    with db.csv_lig(write=False) as df:
        names = [f'worker{i}_{j}' for i in range(N_WORKERS) for j in range(N_UPDATES)]
        new = df.loc[[(name, 'O4') for name in names]]
        ref = [100.0 * i + j for i in range(N_WORKERS) for j in range(N_UPDATES)]
        np.testing.assert_array_equal(new[V_BULK], ref)

        # Every ligand has its own structure and properties in the .hdf5 file
        hdf5_index = new[HDF5_INDEX].to_numpy()
        assert len(set(hdf5_index)) == len(names)
    with db.hdf5('r') as f:
        np.testing.assert_array_equal(f['ligand/properties/V_bulk'][np.sort(hdf5_index)],
                                      new[V_BULK].to_numpy()[np.argsort(hdf5_index)])


def test_threads(tmp_path: Path) -> None:
    """The lock is reentrant within a thread and exclusive between threads."""
    lock = get_lock(tmp_path)
    assert lock is get_lock(str(tmp_path))
    assert Path(lock.lock_file) == tmp_path / LOCK_NAME

    events = []
    with lock:
        with lock:  # Reentrant
            thread = threading.Thread(target=lambda: (lock.acquire(), events.append('thread'),
                                                      lock.release()))
            thread.start()
            thread.join(0.2)
            assert thread.is_alive() and not events  # Waits for this thread
        events.append('main')
    thread.join(5)
    assert events == ['main', 'thread']


def test_open_hdf5(tmp_path: Path) -> None:
    """:func:`open_hdf5` holds the lock while the file is open."""
    filename = tmp_path / 'test.hdf5'
    lock = get_lock(tmp_path)
    with open_hdf5(filename, 'a') as f:
        f['a'] = 1
        assert lock.is_locked
    assert not lock.is_locked
    with open_hdf5(filename) as f:
        assert f['a'][()] == 1
