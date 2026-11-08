"""Conceptual DFT (CDFT) descriptors of ligands, calculated with ADF or xTB."""

from math import nan
from typing import Any
from collections.abc import Iterable

import yaml
import numpy as np
import pandas as pd

from qmflows import templates as _templates
from qmflows.packages import ADF_Result
from scm.plams import Molecule, Settings, Units, Results
from scm.plams.core.basejob import Job
from CAT._compat import ADFJob
from CAT.workflows import (
    WorkFlow, JOB_SETTINGS_CDFT, JOB_SETTINGS_CDFT_XTB, MOL, CDFT_XTB_COLUMNS
)
from CAT.jobs import job_single_point  # noqa: F401
from CAT.settings_dataframe import SettingsDataFrame
from CAT.frozen_settings import thaw_settings
from CAT.xtb import TBLiteJob, check_tblite

__all__ = ['init_cdft', 'get_global_descriptors', 'get_xtb_descriptors', 'cdft']

_CDFT: str = """specific:
    adf:
        symmetry: nosym
        conceptualdft:
            enabled: yes
            analysislevel: extended
            electronegativity: yes
            domains:
                enabled: yes
        qtaim:
            enabled: yes
            analysislevel: extended
            energy: yes
        basis:
            core: none
            type: DZP
        xc:
            libxc:
                CAM-B3LYP
        numericalquality: good
"""

#: A QMFlows-style template for conceptual DFT calculations.
cdft = Settings(yaml.safe_load(_CDFT))
cdft.specific.adf += _templates.singlepoint.specific.adf.copy()


def init_cdft(ligand_df: SettingsDataFrame) -> None:
    r"""Initialize the ligand conceptual dft (CDFT) workflow.

    The descriptors are computed with ADF (``job1: ADFJob``, the default) or,
    if ``job1`` is :class:`~CAT.xtb.TBLiteJob`, from the vertical ionization potential and
    electron affinity computed with xTB (see :func:`get_xtb_descriptors`).
    The latter are stored in the ``cdft_xtb`` columns.

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|
        A DataFrame of ligands.

    """
    job1 = ligand_df.settings.optional.ligand.cdft.job1
    xtb = isinstance(job1, type) and issubclass(job1, TBLiteJob)
    if xtb:
        check_tblite()
    workflow = WorkFlow.from_template(ligand_df, name='cdft_xtb' if xtb else 'cdft')

    # Import from the database and start the calculation
    idx = workflow.from_db(ligand_df)
    workflow(start_xtb_jobs if xtb else start_crs_jobs, ligand_df, index=idx)

    # Sets a nested list with the filenames of .in files
    # This cannot be done with loc is it will try to expand the list into a 2D array
    job_settings = JOB_SETTINGS_CDFT_XTB if xtb else JOB_SETTINGS_CDFT
    ligand_df[job_settings] = workflow.pop_job_settings(ligand_df[MOL])

    # Export to the database
    job_recipe = workflow.get_recipe()
    workflow.to_db(ligand_df, index=idx, job_recipe=job_recipe)


def start_crs_jobs(mol_list: Iterable[Molecule],
                   jobs: Iterable[type[Job]], settings: Iterable[Settings],
                   **kwargs: Any) -> list[pd.Series]:
    # Parse the job type
    job, *_ = jobs
    if ADFJob is None:
        raise NotImplementedError("The conceptual DFT workflow requires the ADFJob of PLAMS < 2024 "
                                  "or, for xTB-based descriptors, `job1: TBLiteJob`")
    elif job is not ADFJob:
        raise NotImplementedError(f"job: {job.__class__.__name__} = {job!r}")

    # Parse and update the input settings
    _s, *_ = settings
    s = Settings(_s)
    s.input += cdft.specific.adf

    ret = []
    for mol in mol_list:
        ret.append(run_cdft_job(mol, job, s))
    return ret


#: The default method of the xTB-based conceptual DFT descriptors:
#: IPEA1-xTB, parametrized for ionization potentials and electron affinities
XTB_METHOD = "IPEA1-xTB"


def start_xtb_jobs(mol_list: Iterable[Molecule],
                   jobs: Iterable[type[Job]], settings: Iterable[Settings],
                   **kwargs: Any) -> np.ndarray:
    """Compute the xTB-based conceptual DFT descriptors of all molecules in **mol_list**.

    Returns a 2D array with the descriptors of :data:`CAT.workflows.CDFT_XTB_COLUMNS`
    for each molecule.

    """
    _s, *_ = settings
    s = thaw_settings(_s) if _s is not None else Settings()
    s.input.task = "ipea"
    s.input.method = s.input.get("method", XTB_METHOD)

    ret = [run_xtb_job(mol, s) for mol in mol_list]
    return np.array(ret, dtype=np.float64).reshape(-1, len(CDFT_XTB_COLUMNS))


def run_xtb_job(mol: Molecule, s: Settings) -> pd.Series:
    """Run a vertical IP/EA calculation with xTB and return the conceptual DFT descriptors."""
    results = mol.job_single_point(TBLiteJob, s.copy(), name='CDFT_xTB',
                                   ret_results=True, read_template=False)
    if results.job.status in {'crashed', 'failed'}:
        return get_xtb_descriptors(nan, nan)
    return get_xtb_descriptors(
        results.get_ionization_potential("eV"), results.get_electron_affinity("eV")
    )


def get_xtb_descriptors(ip: float, ea: float) -> pd.Series:
    r"""Compute the global conceptual DFT descriptors from the ionization potential and electron affinity.

    The finite-difference (:math:`\Delta`\ SCF) expressions, with the hardness
    :math:`\eta = I - A` as in the conceptual DFT module of ADF:

    ==================================== =================================================
    Electronic chemical potential        :math:`\mu = -(I + A) / 2`
    Electronegativity                    :math:`\chi = -\mu`
    Hardness                             :math:`\eta = I - A`
    Softness                             :math:`S = 1 / \eta`
    Electrophilicity index               :math:`\omega = \mu^2 / 2\eta`
    Dissociation energy (nucleofuge)     :math:`\omega - A = (I - 3A)^2 / 8(I - A)`
    Dissociation energy (electrofuge)    :math:`\omega + I = (3I - A)^2 / 8(I - A)`
    Electrodonating power                :math:`\omega^- = (3I + A)^2 / 16(I - A)`
    Electroaccepting power               :math:`\omega^+ = (I + 3A)^2 / 16(I - A)`
    Net electrophilicity                 :math:`\omega^+ + \omega^-`
    Electronic chemical potential (mu-)  :math:`\mu^- = -I`
    Electronic chemical potential (mu+)  :math:`\mu^+ = -A`
    ==================================== =================================================

    Parameters
    ----------
    ip : :class:`float`
        The vertical ionization potential :math:`I`, in eV.
    ea : :class:`float`
        The vertical electron affinity :math:`A`, in eV.

    Returns
    -------
    :class:`pandas.Series`
        The descriptors in eV (the softness in 1/eV), with the
        :data:`CAT.workflows.CDFT_XTB_COLUMNS` as index.
        The descriptors that depend on :math:`\eta` are ``nan`` if :math:`I \le A`.

    """  # noqa: E501
    eta = ip - ea if ip > ea else nan
    mu = -(ip + ea) / 2
    omega = mu**2 / (2 * eta)
    w_minus = (3 * ip + ea)**2 / (16 * eta)
    w_plus = (ip + 3 * ea)**2 / (16 * eta)
    values = [
        ip, ea, mu, -mu, eta, 1 / eta, omega, omega - ea, omega + ip,
        w_minus, w_plus, w_minus + w_plus, -ip, -ea,
    ]
    index = pd.MultiIndex.from_tuples(CDFT_XTB_COLUMNS, names=['index', 'sub index'])
    return pd.Series(values, index=index, dtype=np.float64, name='global descriptors')


_BACKUP = pd.Series({
    'Electronic chemical potential (mu)': nan,
    'Electronegativity (chi=-mu)': nan,
    'Hardness (eta)': nan,
    'Softness (S)': nan,
    'Hyperhardness (gamma)': nan,
    'Electrophilicity index (w=omega)': nan,
    'Dissocation energy (nucleofuge)': nan,
    'Dissociation energy (electrofuge)': nan,
    'Electrodonating power (w-)': nan,
    'Electroaccepting power(w+)': nan,
    'Net Electrophilicity': nan,
    'Global Dual Descriptor Deltaf+': nan,
    'Global Dual Descriptor Deltaf-': nan,
    'Electronic chemical potential (mu+)': nan,
    'Electronic chemical potential (mu-)': nan
})
_BACKUP.index = pd.MultiIndex.from_product(
    [['cdft'], _BACKUP.index], names=['index', 'sub index']
)


def run_cdft_job(mol: Molecule, job: type[Job], s: Settings) -> pd.Series:
    """Run a conceptual DFT job and extract & return all global descriptors."""
    results = mol.job_single_point(job, s.copy(), name='CDFT',
                                   ret_results=True, read_template=False)

    if results.job.status in {'crashed', 'failed'}:
        return _BACKUP

    ret = get_global_descriptors(results)
    ret.index = pd.MultiIndex.from_product(
        [['cdft'], ret.index], names=['index', 'sub index']
    )
    return ret


def get_global_descriptors(results: Results | ADF_Result) -> pd.Series:
    """Extract a dictionary with all ADF conceptual DFT global descriptors from **results**.

    Examples
    --------
    .. code:: python

        >>> import pandas as pd
        >>> from scm.plams import ADFResults
        >>> from CAT.recipes import get_global_descriptors

        >>> results = ADFResults(...)

        >>> series: pd.Series = get_global_descriptors(results)
        >>> print(dct)
        Electronic chemical potential (mu)     -0.113
        Electronegativity (chi=-mu)             0.113
        Hardness (eta)                          0.090
        Softness (S)                           11.154
        Hyperhardness (gamma)                  -0.161
        Electrophilicity index (w=omega)        0.071
        Dissocation energy (nucleofuge)         0.084
        Dissociation energy (electrofuge)       6.243
        Electrodonating power (w-)              0.205
        Electroaccepting power(w+)              0.092
        Net Electrophilicity                    0.297
        Global Dual Descriptor Deltaf+          0.297
        Global Dual Descriptor Deltaf-         -0.297
        Electronic chemical potential (mu+)    -0.068
        Electronic chemical potential (mu-)    -0.158
        Name: global descriptors, dtype: float64


    Parameters
    ----------
    results : :class:`plams.ADFResults` or :class:`qmflows.ADF_Result`
        A PLAMS Results or QMFlows Result instance of an ADF calculation.

    Returns
    -------
    :class:`pandas.Series`
        A Series with all ADF global decsriptors as extracted from **results**.

    """
    if not isinstance(results, Results):
        results = results.results
        if results is None:
            raise TypeError("The QMFlows result does not contain any PLAMS results")
    file = results['$JN.out']

    with open(file) as f:
        # Identify the GLOBAL DESCRIPTORS block
        for item in f:
            if item == ' GLOBAL DESCRIPTORS\n':
                next(f)
                next(f)
                break
        else:
            raise ValueError(f"Failed to identify the 'GLOBAL DESCRIPTORS' block in {file!r}")

        # Extract the descriptors
        ret = {}
        for item in f:
            item = item.rstrip('\n')
            if not item:
                break

            _key, _value = item.rsplit('=', maxsplit=1)
            key = _key.strip()
            try:
                value = float(_value)
            except ValueError:
                value = float(_value.rstrip('(eV)')) * Units.conversion_ratio('ev', 'au')
            ret[key] = value

    # Fix the names of "mu+" and "mu-"
    ret['Electronic chemical potential (mu+)'] = ret.pop('mu+', nan)
    ret['Electronic chemical potential (mu-)'] = ret.pop('mu-', nan)
    return pd.Series(ret, name='global descriptors')
