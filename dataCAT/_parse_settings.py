
"""A private helper module for parsing job settings .inp files."""

from collections.abc import Sequence
from itertools import count

import h5py
import numpy as np
import pandas as pd

from nanoutils import PathType
from CAT.workflows import HDF5_INDEX

__all__: list[str] = []


def _update_hdf5_settings(f: h5py.File, df: pd.DataFrame, column: str) -> None:
    """Export all files in **df[column]** to hdf5 dataset **column**."""
    i, j, k = f[column].shape

    # Create a 3D array of input files
    try:
        job_ar = _read_inp(df[column], j, k)
    except ValueError:  # df[column] consists of empty lists, abort
        return None

    # Reshape **self.hdf5**
    k = max(i, 1 + int(df[HDF5_INDEX].max()))
    f[column].shape = k, job_ar.shape[1], job_ar.shape[2]

    # Update the hdf5 dataset
    idx = df[HDF5_INDEX].values.astype(int, copy=False)
    idx_argsort = np.argsort(idx)
    f[column][idx[idx_argsort]] = job_ar[idx_argsort]


def _read_inp(job_paths: Sequence[str], ax2: int = 0, ax3: int = 0) -> np.ndarray:
    """Convert all files in **job_paths** (nested sequence of filenames) into a 3D array."""
    # Determine the minimum size of the to-be returned 3D array
    line_count = [[_get_line_count(j) for j in i] for i in job_paths]
    ax1 = len(line_count)
    ax2 = max(ax2, max(len(i) for i in line_count))
    ax3 = max(ax3, max(j for i in line_count for j in i))

    # Create and return a padded 3D array of strings
    ret = np.zeros((ax1, ax2, ax3), dtype='S120')
    for i, list1, list2 in zip(count(), line_count, job_paths):
        for j, k, filename in zip(count(), list1, list2):
            with open(filename, "rb") as f:
                iterator = (i.strip(b"\n") for i in f)
                ret[i, j, :k] = [i for i in iterator if i]
    return ret


def _get_line_count(filename: PathType) -> int:
    """Return the total number of lines in **filename**."""
    with open(filename) as f:
        return sum(line != '\n' for line in f)
