"""A module with various .pdb files used by :mod:`dataCAT.testing_utils`.

Index
-----
.. currentmodule:: dataCAT.data
.. autosummary::
    PDB_TUPLE

API
---
.. autodata:: PDB_TUPLE
    :annotation: : tuple[str, ...] = ...

"""

import os

_DATA = os.path.abspath(os.path.dirname(__file__))

#: A tuple with the absolute paths to all .pdb files in :mod:`dataCAT.data`.
PDB_TUPLE: tuple[str, ...] = tuple(sorted(
    os.path.join(_DATA, f) for f in os.listdir(_DATA) if f.endswith('pdb')
))
del _DATA
del os

__all__ = ['PDB_TUPLE']
