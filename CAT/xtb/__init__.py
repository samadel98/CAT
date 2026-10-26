"""Extended tight-binding (xTB) calculations with tblite.

:class:`TBLiteJob` runs GFN2-xTB (or GFN1-xTB) single points, geometry optimizations
and frequency analyses with `tblite <https://github.com/tblite/tblite>`_,
which is available on PyPI (``pip install tblite``).
It can be used wherever CAT accepts a PLAMS job type, with the ``"tblite"`` alias.

Index
-----
.. currentmodule:: CAT.xtb
.. autosummary::
    TBLiteJob
    TBLiteResults
    check_tblite

API
---
.. autoclass:: TBLiteJob
    :members: get_input, get_runscript, check
.. autoclass:: TBLiteResults
    :members:
.. autofunction:: check_tblite

"""

from ._job import TBLiteJob, TBLiteResults, TEMPLATES, check_tblite

__all__ = ["TBLiteJob", "TBLiteResults", "TEMPLATES", "check_tblite"]
