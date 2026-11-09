"""Conformer searches of ligands: an RDKit conformer ensemble, optimized and ranked with xTB.

Conformer search
================

For example, the conformers of hexanoate, optimized with GFN2-xTB:

.. code:: python

    >>> from scm.plams import from_smiles, init, finish
    >>> from nanoCAT.conformers import conformer_search

    >>> init()
    >>> mol = from_smiles("CCCCCC(=O)[O-]")
    >>> conformers = conformer_search(mol, n=20)
    >>> finish()
    >>> for conf in conformers:
    ...     print(conf.properties.conformer, round(conf.properties.energy.E_rel, 2))
    2 0.0
    1 0.09
    ...

Index
-----
.. currentmodule:: nanoCAT.conformers
.. autosummary::
    conformer_search
    init_conformers
    start_conformer_jobs

API
---
.. autofunction:: conformer_search
.. autofunction:: init_conformers
.. autofunction:: start_conformer_jobs

"""

from __future__ import annotations

import os
import itertools
from typing import Any
from collections.abc import Iterable, Iterator

import numpy as np
from rdkit.Chem import rdDistGeom
from scm.plams import Molecule, Settings
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

from CAT.jobs import job_geometry_opt  # noqa: F401
from CAT.logger import logger
from CAT.parallel import thread_map
from CAT.workflows import WorkFlow, MOL, JOB_SETTINGS_CONFORMERS
from CAT.settings_dataframe import SettingsDataFrame
from CAT.frozen_settings import thaw_settings
from CAT.xtb import TBLiteJob

__all__ = ['conformer_search', 'init_conformers', 'start_conformer_jobs']

#: Two optimized conformers are considered identical if their energies differ by less than
#: this value (kcal/mol) and their principal moments of inertia by less than
#: :data:`MOMENT_RTOL` (relative), like the conformer sorting of CREST
ENERGY_ATOL = 0.1

#: See :data:`ENERGY_ATOL`
MOMENT_RTOL = 0.01

_COUNTER = itertools.count(1)


def _embed(mol: Molecule, n: int, seed: int, prune_rms: float = 0.5) -> list[np.ndarray]:
    """Return the coordinates of up to **n** conformers of **mol**, generated with RDKit ETKDG."""
    rdmol = molkit.to_rdmol(mol, properties=False)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed  # type: ignore[assignment]  # Incorrect RDKit stubs
    params.pruneRmsThresh = prune_rms  # type: ignore[assignment]
    try:
        conf_ids = list(rdDistGeom.EmbedMultipleConfs(rdmol, numConfs=n, params=params))
    except Exception as ex:
        logger.warning(f"{mol.properties.get('name', 'molecule')}: failed to generate "
                       f"conformers with RDKit: {ex!r}")
        return []
    return [rdmol.GetConformer(i).GetPositions() for i in conf_ids]


def _moments(mol: Molecule) -> np.ndarray:
    """Return the principal moments of inertia of **mol**."""
    xyz = mol.as_array()
    masses = np.array([at.mass for at in mol])
    xyz = xyz - masses @ xyz / masses.sum()
    inertia = np.einsum('i,ij,ik->jk', masses, xyz, xyz)
    return np.linalg.eigvalsh(np.trace(inertia) * np.eye(3) - inertia)


def _unique(mol_list: Iterable[Molecule]) -> Iterator[Molecule]:
    """Yield the molecules of the energy-sorted **mol_list** that are not duplicates."""
    kept: list[tuple[float, np.ndarray]] = []
    for mol in mol_list:
        energy = mol.properties.energy.E
        moments = _moments(mol)
        for e, m in kept:
            if abs(energy - e) < ENERGY_ATOL and np.allclose(moments, m, rtol=MOMENT_RTOL):
                break
        else:
            kept.append((energy, moments))
            yield mol


def _optimize(mol: Molecule, job: type[Job], settings: Settings, name: str) -> None:
    mol.job_geometry_opt(job, thaw_settings(settings), name=name, read_template=True)


def conformer_search(mol: Molecule, n: int = 10, job: type[Job] = TBLiteJob,
                     settings: Settings | None = None, seed: int = 1,
                     parallel_jobs: int = 1, unique: bool = True) -> list[Molecule]:
    r"""Return the unique conformers of **mol**, optimized with **job** and sorted by energy.

    The conformers are the geometry of **mol** itself and up to **n** conformers
    generated with RDKit (ETKDG), which are all optimized with **job**, by default
    GFN2-xTB (:class:`~CAT.xtb.TBLiteJob`).
    Conformers whose energies differ by less than 0.1 kcal/mol and whose principal moments of
    inertia differ by less than 1% are considered identical.

    PLAMS must be initialized (:func:`scm.plams.init`).

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The molecule, with explicit hydrogens; it is not modified.
    n : :class:`int`
        The number of conformers generated with RDKit.
    job : :class:`type[Job] <type>`
        The job type of the geometry optimizations, *e.g.* :class:`~CAT.xtb.TBLiteJob` or
        :class:`~CAT.ase.ASEJob`.
    settings : :class:`~scm.plams.core.settings.Settings`, optional
        The settings of the geometry optimizations.
    seed : :class:`int`
        The random seed of RDKit.
    parallel_jobs : :class:`int`
        The number of geometry optimizations that run at the same time.
    unique : :class:`bool`
        Whether to remove duplicate conformers.

    Returns
    -------
    :class:`list[Molecule] <list>`
        The optimized conformers, lowest energy first; failed optimizations are left out.
        Their energies (kcal/mol) are stored in ``properties.energy.E``,
        the energies relative to the first one in ``properties.energy.E_rel``,
        and ``properties.conformer`` is the index of their initial geometry:
        ``0`` for the geometry of **mol** and :math:`i > 0` for the RDKit conformers.

    """
    s = Settings() if settings is None else settings

    # Skip identical geometries, e.g. an RDKit conformer identical to the input geometry
    # if the latter was generated by RDKit with the same seed
    xyz_list = [mol.as_array()]
    for xyz in _embed(mol, n, seed):
        if not any(np.allclose(xyz, xyz_old, rtol=0, atol=1e-4) for xyz_old in xyz_list):
            xyz_list.append(xyz)

    conformers = []
    for i, xyz in enumerate(xyz_list):
        conf = mol.copy()
        conf.from_array(xyz)
        conf.properties.job_path = []
        conf.properties.is_opt = False
        conf.properties.conformer = i
        conformers.append(conf)

    args = [(conf, job, s, f'conformer_{next(_COUNTER)}') for conf in conformers]
    thread_map(_optimize, args, parallel_jobs)

    job_paths = [path for conf in conformers for path in conf.properties.job_path]
    opt = [conf for conf in conformers if conf.properties.is_opt is True
           and np.isfinite(conf.properties.energy.get('E', np.nan))]
    opt.sort(key=lambda m: m.properties.energy.E)
    ret = list(_unique(opt)) if unique else opt
    for conf in ret:
        conf.properties.energy.E_rel = conf.properties.energy.E - ret[0].properties.energy.E
        conf.properties.job_path = job_paths
    return ret


def _write_xyz(filename: str | os.PathLike[str], mol_list: Iterable[Molecule]) -> None:
    """Write the conformers to a multi-frame .xyz file."""
    with open(filename, 'w') as f:
        for mol in mol_list:
            e = mol.properties.energy
            f.write(f'{len(mol)}\nconformer {mol.properties.conformer}: '
                    f'E = {e.E:.4f} kcal/mol, E_rel = {e.E_rel:.4f} kcal/mol\n')
            for at in mol:
                x, y, z = at.coords
                f.write(f'{at.symbol:<3} {x:16.10f} {y:16.10f} {z:16.10f}\n')


def init_conformers(ligand_df: SettingsDataFrame) -> None:
    """Run a conformer search of each ligand (``optional.ligand.conformers``).

    Performs an inplace update of **ligand_df**, adding the ``("conformers", "count")`` and
    ``("conformers", "E_rel")`` columns: the number of unique conformers and the energy
    of the conformer of CAT (*i.e.* the one attached to the core, after its optimization
    with the job of the conformer search) relative to the lowest-energy conformer (kcal/mol).
    The conformers are written, lowest energy first, to ``{ligand name}.conformers.xyz``
    in the ligand directory.

    """
    workflow = WorkFlow.from_template(ligand_df, name='conformers')
    idx = workflow.from_db(ligand_df)
    workflow(start_conformer_jobs, ligand_df, index=idx)

    ligand_df[JOB_SETTINGS_CONFORMERS] = workflow.pop_job_settings(ligand_df[MOL])
    job_recipe = workflow.get_recipe()
    workflow.to_db(ligand_df, index=idx, job_recipe=job_recipe)


def start_conformer_jobs(mol_list: Iterable[Molecule],
                         jobs: tuple[type[Job], ...], settings: tuple[Settings | None, ...],
                         n_conformers: int = 10, seed: int = 1, parallel_jobs: int = 1,
                         path: str | os.PathLike[str] = '.', **kwargs: Any) -> np.ndarray:
    """Run the conformer search of all ligands in **mol_list**.

    Returns a 2D array with the number of unique conformers and the relative energy
    (kcal/mol) of the input conformer of each ligand.

    """
    job, *_ = jobs
    s, *_ = settings
    ret = []
    for ligand in mol_list:
        name = ligand.properties.name
        logger.info(f'{name}: conformer search with {n_conformers} RDKit conformers '
                    f'and {job.__name__}')
        opt = conformer_search(ligand, n=n_conformers, job=job, settings=s,
                               seed=seed, parallel_jobs=parallel_jobs, unique=False)
        ligand.properties.job_path = opt[0].properties.job_path if opt else []
        if not opt:
            logger.error(f'{name}: all conformer optimizations failed')
            ret.append([np.nan, np.nan])
            continue

        # The energy of the input conformer, even if it is a duplicate of another one
        conformers = list(_unique(opt))
        _write_xyz(os.path.join(path, f'{name}.conformers.xyz'), conformers)
        e_rel = next((c.properties.energy.E_rel for c in opt if c.properties.conformer == 0),
                     np.nan)
        ret.append([len(conformers), e_rel])
        logger.info(f'{name}: {len(conformers)} unique conformers; the input conformer is '
                    f'{e_rel:.2f} kcal/mol above the lowest one')
    return np.array(ret, dtype=np.float64).reshape(-1, 2)
