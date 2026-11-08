"""Ligand solvation free energies computed with xTB and an implicit solvation model.

Index
-----
.. currentmodule:: nanoCAT.xtb_solvation
.. autosummary::
    init_xtb_solv
    start_xtb_solv_jobs

API
---
.. autofunction:: init_xtb_solv
.. autofunction:: start_xtb_solv_jobs

"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Sequence

import numpy as np

from scm.plams import Molecule, Settings
from scm.plams.core.basejob import Job

from CAT.jobs import job_single_point  # noqa: F401
from CAT.workflows import WorkFlow, MOL, JOB_SETTINGS_SOLV_XTB
from CAT.settings_dataframe import SettingsDataFrame
from CAT.frozen_settings import thaw_settings
from CAT.xtb import TBLiteJob, check_tblite

__all__ = ['init_xtb_solv', 'start_xtb_solv_jobs']

#: The first level of the columns with the solvation free energies (kcal/mol)
SOLV_COLUMN = 'E_solv_xtb'


def init_xtb_solv(ligand_df: SettingsDataFrame) -> None:
    """Compute the solvation free energy of each ligand in the ``optional.ligand.solvation`` solvents.

    Performs an inplace update of **ligand_df**, adding an ``("E_solv_xtb", solvent)``
    column (kcal/mol) for each solvent.

    The solvation free energy is the difference between the xTB energies of the ligand with
    and without the implicit solvation model (ALPB by default), at the geometry of the ligand.

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|
        A DataFrame of ligands.

    """  # noqa: E501
    check_tblite()
    workflow = WorkFlow.from_template(ligand_df, name='solv_xtb')

    columns = [(SOLV_COLUMN, solvent) for solvent in workflow.solvents]
    import_columns = dict.fromkeys(columns, np.nan)
    export_columns = columns + list(workflow.export_columns)

    idx = workflow.from_db(ligand_df, columns=import_columns)
    workflow(start_xtb_solv_jobs, ligand_df, index=idx, columns=columns)

    job_recipe = workflow.get_recipe()
    ligand_df[JOB_SETTINGS_SOLV_XTB] = workflow.pop_job_settings(ligand_df[MOL])
    workflow.to_db(ligand_df, index=idx, columns=export_columns, job_recipe=job_recipe)


def start_xtb_solv_jobs(mol_list: Iterable[Molecule],
                        jobs: tuple[type[Job], ...], settings: tuple[Settings | None, ...],
                        solvents: Sequence[str] = (), **kwargs: Any) -> np.ndarray:
    """Compute the solvation free energies (kcal/mol) of all molecules in **mol_list**.

    Returns a 2D array with a row for each molecule and a column for each of the **solvents**.

    """
    _s, *_ = settings
    s = thaw_settings(_s) if _s is not None else Settings()
    s.input.task = 'solvation'
    s.input.solvents = list(solvents)

    ret = []
    for mol in mol_list:
        mol.properties.job_path = []
        results = mol.job_single_point(TBLiteJob, s.copy(), name='solvation_xTB',
                                       ret_results=True, read_template=False)
        if results.job.status in {'crashed', 'failed'}:
            ret.append([np.nan] * len(solvents))
            continue
        energies = results.get_solvation_energies('kcal/mol')
        ret.append([energies[solvent] for solvent in solvents])
    return np.array(ret, dtype=np.float64).reshape(-1, len(solvents))
