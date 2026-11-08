"""Calculations with any `ASE <https://ase-lib.org>`_ calculator.

:class:`ASEJob` runs single points, geometry optimizations and frequency analyses with
an ASE calculator, *e.g.* a machine-learned interatomic potential such as MACE.
It can be used wherever CAT accepts a PLAMS job type, with the ``"ase"`` alias.

Index
-----
.. currentmodule:: CAT.ase
.. autosummary::
    ASEJob
    ASEResults
    check_ase

API
---
.. autoclass:: ASEJob
    :members: get_input
.. autoclass:: ASEResults
.. autofunction:: check_ase

"""

from ._job import ASEJob, ASEResults, check_ase

__all__ = ["ASEJob", "ASEResults", "check_ase"]
