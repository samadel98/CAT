"""Run the CPU-bound parts of CAT's workflows in parallel processes.

Index
-----
.. currentmodule:: CAT.parallel
.. autosummary::
    get_processes
    parallel_map
    pack_atom_refs
    unpack_atom_refs

API
---
.. autofunction:: get_processes
.. autofunction:: parallel_map
.. autofunction:: pack_atom_refs
.. autofunction:: unpack_atom_refs

"""

from __future__ import annotations

import os
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from scm.plams import Molecule, Atom

from .logger import logger

__all__ = ['get_processes', 'parallel_map', 'pack_atom_refs', 'unpack_atom_refs']

T = TypeVar('T')

#: Environment variables limiting the number of threads of the numerical libraries;
#: every worker process gets one thread, so that the workers do not compete for the cores
_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def get_processes(value: None | int) -> int:
    """Return the number of processes for the ``optional.processes`` value **value**.

    ``None`` and ``0`` mean "one per CPU core".

    Examples
    --------
    .. code:: python

        >>> import os
        >>> from CAT.parallel import get_processes

        >>> get_processes(4)
        4
        >>> get_processes(0) == (os.cpu_count() or 1)
        True

    """
    if not value:
        return os.cpu_count() or 1
    return int(value)


def parallel_map(func: Callable[..., T], args: Iterable[tuple[Any, ...]],
                 processes: int = 1) -> list[T]:
    """Return ``[func(*a) for a in args]``, computed by **processes** worker processes.

    With ``processes=1`` (or a single item) everything runs in the current process.
    Otherwise **func** and its arguments must be picklable, and **func** must be importable
    by the (``"spawn"``-ed) worker processes, *i.e.* a module-level function.

    Note
    ----
    As the workers are spawned, scripts that call CAT with ``optional.processes > 1``
    must guard their entry point with ``if __name__ == "__main__":``.
    If the worker processes cannot be started or die, everything is computed in the current
    process instead (with a warning).

    """
    arg_list = list(args)
    n = min(processes, len(arg_list))
    if n <= 1:
        return [func(*a) for a in arg_list]

    chunksize = max(1, math.ceil(len(arg_list) / (4 * n)))
    ctx = multiprocessing.get_context('spawn')
    try:
        with (
            _single_threaded_children(),
            ProcessPoolExecutor(max_workers=n, mp_context=ctx) as executor,
        ):
            return list(executor.map(func, *zip(*arg_list), chunksize=chunksize))
    except BrokenProcessPool as ex:
        logger.warning(
            f"The worker processes failed ({ex}); continuing in a single process. Note that "
            "scripts that run CAT with `optional.processes` > 1 must guard their entry point "
            "with `if __name__ == '__main__':`"
        )
        return [func(*a) for a in arg_list]


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Set :data:`_THREAD_VARS` to 1 (unless set by the user) for processes started within."""
    new = [var for var in _THREAD_VARS if var not in os.environ]
    os.environ.update(dict.fromkeys(new, '1'))
    try:
        yield
    finally:
        for var in new:
            os.environ.pop(var, None)


def pack_atom_refs(mol: Molecule) -> dict[str, int | list[int]]:
    """Return the indices of all atoms of **mol** that its properties refer to.

    PLAMS pickles a molecule's properties separately from its atoms, so after
    unpickling, a property such as ``mol.properties.dummies`` no longer refers to an
    atom of the molecule. Use :func:`unpack_atom_refs` to restore these references.

    Examples
    --------
    .. code:: python

        >>> import pickle
        >>> from scm.plams import from_smiles
        >>> from CAT.parallel import pack_atom_refs, unpack_atom_refs

        >>> mol = from_smiles('CCO')
        >>> mol.properties.dummies = mol[3]
        >>> refs = pack_atom_refs(mol)
        >>> refs
        {'dummies': 2}

        >>> mol2 = pickle.loads(pickle.dumps(mol))
        >>> unpack_atom_refs(mol2, refs)
        >>> mol2.properties.dummies is mol2[3]
        True

    """
    idx = {id(at): i for i, at in enumerate(mol.atoms)}
    ret: dict[str, int | list[int]] = {}
    for key, value in mol.properties.items():
        if isinstance(value, Atom) and id(value) in idx:
            ret[key] = idx[id(value)]
        elif (
            isinstance(value, list) and value and
            all(isinstance(at, Atom) and id(at) in idx for at in value)
        ):
            ret[key] = [idx[id(at)] for at in value]
    return ret


def unpack_atom_refs(mol: Molecule, refs: dict[str, int | list[int]]) -> None:
    """Restore the atom references of :func:`pack_atom_refs` in **mol**."""
    for key, value in refs.items():
        if isinstance(value, int):
            mol.properties[key] = mol.atoms[value]
        else:
            mol.properties[key] = [mol.atoms[i] for i in value]
