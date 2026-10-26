"""A module designed for optimizing the combined ligand & core.

Index
-----
.. currentmodule:: CAT.attachment.qd_opt
.. autosummary::
    init_qd_opt
    start_qd_opt
    get_job_settings
    _qd_to_db
    qd_opt

API
---
.. autofunction:: init_qd_opt
.. autofunction:: start_qd_opt
.. autofunction:: get_job_settings
.. autofunction:: _qd_to_db
.. autofunction:: qd_opt

"""

from typing import NoReturn, Any
from collections.abc import Iterable

from scm.plams import Molecule, Settings
from scm.plams.core.basejob import Job

from ..jobs import job_geometry_opt, freeze_atoms  # noqa: F401
from ..workflows import WorkFlow, MOL, JOB_SETTINGS_QD_OPT
from ..mol_utils import fix_carboxyl, fix_h
from ..settings_dataframe import SettingsDataFrame
from ..data_handling.mol_to_file import mol_to_file
from ..logger import logger
from ..parallel import thread_map

try:
    from nanoCAT.qd_opt_ff import qd_opt_ff
except Exception as ex:  # A broken Nano-CAT install should not break `import CAT`
    def qd_opt_ff(*args: Any, ex: Exception = ex, **kwargs: Any) -> NoReturn: raise ex

__all__ = ['init_qd_opt']


def init_qd_opt(qd_df: SettingsDataFrame) -> None:
    """Initialize the ligand optimization procedure."""
    workflow = WorkFlow.from_template(qd_df, name='qd_opt')

    # Pull from the database; push unoptimized structures
    idx = workflow.from_db(qd_df)
    parallel_jobs = qd_df.settings.optional.get('parallel_jobs', 1)
    workflow(start_qd_opt, qd_df, columns=[], index=idx, parallel_jobs=parallel_jobs)

    # Sets a nested list
    # This cannot be done with loc is it will try to expand the list into a 2D array
    qd_df[JOB_SETTINGS_QD_OPT] = workflow.pop_job_settings(qd_df[MOL])

    # Push the optimized structures to the database
    job_recipe = workflow.get_recipe()
    workflow.to_db(qd_df, status='optimized', index=idx, job_recipe=job_recipe)

    # Export ligands to .xyz, .pdb, .mol and/or .mol format
    mol_format = qd_df.settings.optional.database.mol_format
    if mol_format:
        path = workflow.path
        mol_to_file(qd_df.loc[idx, MOL], path, mol_format=mol_format)


def start_qd_opt(mol_list: Iterable[Molecule],
                 jobs: tuple[type[Job] | None, ...], settings: tuple[Settings | None, ...],
                 use_ff: bool = False, parallel_jobs: int = 1, **kwargs) -> None:
    """Loop over all molecules in **mol_list** and perform geometry optimizations.

    Up to **parallel_jobs** quantum dots are optimized at the same time,
    except with the forcefield (**use_ff**).

    """
    mols = list(mol_list)
    for mol in mols:
        mol.properties.job_path = []

    if use_ff or parallel_jobs <= 1 or len(mols) <= 1:
        for mol in mols:
            qd_opt(mol, jobs, settings, use_ff=use_ff)
        return None

    logger.info(f'Optimizing {len(mols)} quantum dots with up to '
                f'{min(parallel_jobs, len(mols))} at the same time')
    # Unique job names, as PLAMS < 2024 does not rename simultaneously registered jobs safely
    args = [(mol, jobs, settings, False, f'_{i}') for i, mol in enumerate(mols, 1)]
    thread_map(qd_opt, args, parallel_jobs)
    return None


def qd_opt(mol: Molecule, jobs: tuple[type[Job] | None, ...],
           settings: tuple[Settings | None, ...], use_ff: bool = False,
           name_suffix: str = '') -> None:
    """Perform an optimization of the quantum dot.

    Performs an inplace update of **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        The to-be optimized molecule.

    job_recipe : |plams.Settings|_
        A Settings instance containing all jon settings.
        Expects 4 keys: ``"job1"``, ``"job2"``, ``"s1"``, ``"s2"``.

    forcefield : bool
        If ``True``, perform the job with CP2K with a user-specified forcefield.

    name_suffix : str
        A suffix for the names of the jobs.

    """
    # Prepare the job settings
    if use_ff:
        qd_opt_ff(mol, jobs, settings)
        return None

    # Expand arguments
    job1, job2 = jobs
    s1, s2 = settings

    # Freeze the core and the ligand anchors
    s1 = freeze_atoms(job1, s1, mol.properties.indices)
    s2 = freeze_atoms(job2, s2, mol.properties.indices)

    # Run the first job and fix broken angles
    mol.job_geometry_opt(job1, s1, name=f'QD_opt_part1{name_suffix}')
    fix_carboxyl(mol)
    fix_h(mol)
    mol.round_coords()

    # Run the second job
    if job2 is not None:
        mol.job_geometry_opt(job2, s2, name=f'QD_opt_part2{name_suffix}')
        mol.round_coords()
    return None
