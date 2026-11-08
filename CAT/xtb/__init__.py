"""Extended tight-binding (xTB) calculations with tblite.

:class:`TBLiteJob` runs GFN2-xTB (or GFN1-xTB) single points, geometry optimizations,
frequency analyses, solvation free energies and vertical ionization potentials and electron
affinities with `tblite <https://github.com/tblite/tblite>`_,
which is available on PyPI (``pip install tblite``).
It can be used wherever CAT accepts a PLAMS job type, with the ``"tblite"`` alias.

Index
-----
.. currentmodule:: CAT.xtb
.. autosummary::
    TBLiteJob
    TBLiteResults
    RunnerJob
    RunnerResults
    check_tblite

API
---
.. autoclass:: TBLiteJob
    :members: get_input, get_runscript, check
.. autoclass:: TBLiteResults
    :members:
    :inherited-members: Results
.. autoclass:: RunnerJob
.. autoclass:: RunnerResults
.. autofunction:: check_tblite

"""

from ._job import TBLiteJob, TBLiteResults, RunnerJob, RunnerResults, TEMPLATES, check_tblite

__all__ = [
    "TBLiteJob", "TBLiteResults", "RunnerJob", "RunnerResults", "TEMPLATES", "check_tblite"
]
