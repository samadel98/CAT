"""A module designed for sanitizing and interpreting the input file.

Index
-----
.. currentmodule:: CAT.data_handling.validate_input
.. autosummary::
    validate_input

API
---
.. autofunction:: validate_input

"""

from os import mkdir
from os.path import (join, isdir)
from typing import Any
from collections.abc import Mapping

from contextlib import nullcontext

import qmflows
from scm.plams import Settings, Cp2kJob

from .validation_schemas import (
    core_schema,
    ligand_schema,
    qd_schema,
    database_schema,
    mongodb_schema,
    bde_schema,
    qd_opt_schema,
    crs_schema,
    asa_schema,
    ligand_opt_schema,
    subset_schema,
    multi_ligand_schema,
    cdft_schema,
    bulkiness_schema,
    cone_angle_schema,
    processes_schema,
    parallel_jobs_schema,
)

from .validate_ff import validate_ff, update_ff_jobs
from .validate_mol import validate_mol
from .anchor_parsing import parse_anchors
from ..utils import validate_path, SetEnviron
from ..parallel import get_processes
from ..logger import logger
from ..xtb import TBLiteJob

try:
    from dataCAT import Database
    DATA_CAT = True
except Exception:  # A broken Data-CAT install should not break `import CAT`
    DATA_CAT = False

__all__ = ['validate_input']


def _validate_multi_lig(s: Settings) -> None:
    """Check that one (and only one!) of ``'f'`` and ``'dummy'`` is specified."""
    f = s.optional.qd.multi_ligand.f

    anchor = s.optional.qd.multi_ligand.anchor
    if anchor is None:
        anchor = s.optional.qd.multi_ligand.dummy
        s.optional.qd.multi_ligand.anchor = anchor
    del s.optional.qd.multi_ligand.dummy

    if f is anchor is None:
        raise ValueError("'.multi_ligand.f' and '.multi_ligand.anchor' cannot be "
                         "both unspecified or set to 'None'")
    elif None not in (f, anchor):
        raise ValueError("Only one of '.multi_ligand.f' and '.multi_ligand.anchor' "
                         "should be specified")

    if anchor is not None:
        assert len(anchor) == len(s.optional.qd.multi_ligand.ligands)
    else:
        assert len(f) == len(s.optional.qd.multi_ligand.ligands) - 1


def parse_qmflows_keywords(bde_dict: dict[str, Any]) -> dict[str, Any]:
    """Parse qmflows generic keywords for the BDE workflow."""
    # TODO: Support QM packages other than qmflows
    if bde_dict["job1"] is Cp2kJob:
        s1 = qmflows.Settings(bde_dict["s1"])
        s1 = qmflows.cp2k.generic2specific(s1)
        if "specific" in s1:
            s1.input.update(s1.pop("specific").pop(qmflows.cp2k.pkg_name))
        bde_dict["s1"] = s1

    if bde_dict["job2"] is Cp2kJob:
        s2 = qmflows.Settings(bde_dict["s2"])
        s2 = qmflows.cp2k.generic2specific(s2)
        if "specific" in s1:
            s2.input.update(s2.pop("specific").pop(qmflows.cp2k.pkg_name))
        bde_dict["s2"] = s2
    return bde_dict


def tblite_settings(raw: Mapping[str, Any], validated: dict[str, Any]) -> dict[str, Any]:
    """Give a :class:`~CAT.xtb.TBLiteJob` without user-specified settings empty settings.

    The default settings of ``s1`` and ``s2`` are meant for an
    :class:`~scm.plams.interfaces.adfsuite.ams.AMSJob` (or are ``None``).
    They are replaced by empty settings if ``job1`` or ``job2`` is a TBLiteJob,
    while the matching settings are absent from the unvalidated block, **raw**.

    """
    for job_key, s_key in [("job1", "s1"), ("job2", "s2")]:
        if validated.get(job_key) is TBLiteJob and s_key not in raw:
            validated[s_key] = Settings()
    return validated


def validate_input(s: Settings, validate_only: bool = True) -> None:
    """Initialize the input-validation procedure.

    performs an inplace update of **s**.

    Parameters
    ----------
    s : |plams.Settings|_
        A Settings instance with to-be validated CAT input settings.
    validate_only : bool
        Perform only validation.

    """
    dirnames = ('database', 'ligand', 'core', 'qd')
    if not validate_only:
        # Validate the path
        s.path = validate_path(s.get("path"))

        # Set the various working directories
        for key in dirnames:
            value = join(s.path, key)
            s.optional[key].dirname = value
            if not isdir(value):
                mkdir(value)
    else:
        if s.path is None:
            s.path = '.'
        for key in dirnames:
            s.optional[key].dirname = join(s.path, key)

    # Validate optional argument
    if not validate_only:
        context = nullcontext()
    else:
        context = SetEnviron(AMSBIN='', AMSHOME='', AMSRESOURCES='', SCMLICENSE='')

    s.optional.database = database_schema.validate(s.optional.database)
    s.optional.processes = get_processes(processes_schema.validate(s.optional.get('processes', 1)))
    s.optional.parallel_jobs = get_processes(
        parallel_jobs_schema.validate(s.optional.get('parallel_jobs', 1))
    )
    with context:
        s.optional.ligand = ligand_schema.validate(s.optional.ligand)
        s.optional.core = core_schema.validate(s.optional.core)
        s.optional.qd = qd_schema.validate(s.optional.qd)

        # Validate some of the more complex optionala rguments
        if s.optional.database.mongodb:
            s.optional.database.mongodb = mongodb_schema.validate(s.optional.database.mongodb)

        if s.optional.core.subset:
            s.optional.core.subset = subset_schema.validate(s.optional.core.subset)
            if 'p' in s.optional.core.subset:
                if 'weight' in s.optional.core.subset:
                    raise KeyError("'p' and 'weight' cannot be simultaneously specified")
                logger.warn("The 'subset.p' parameter is deprecated; see 'subset.weight'")
                p = s.optional.core.subset.pop('p')
                s.optional.core.subset.weight = lambda x: -(x**p)
        if s.optional.core.anchor is not None:
            pass
        elif s.optional.core.dummy is not None:
            s.optional.core.anchor = s.optional.core.dummy
        else:
            s.optional.core.anchor = 17
        del s.optional.core.dummy

        if s.optional.ligand.optimize:
            s.optional.ligand.optimize = ligand_opt_schema.validate(s.optional.ligand.optimize)
        if s.optional.ligand.cdft:
            s.optional.ligand.cdft = cdft_schema.validate(s.optional.ligand.cdft)

        if s.optional.get('crs') is not None:
            s.optional.ligand.crs = crs_schema.validate(s.optional.crs)
        elif s.optional.ligand['cosmo-rs']:
            crs = s.optional.ligand.pop('cosmo-rs')
            s.optional.ligand.crs = crs_schema.validate(crs)
        if s.optional.ligand.cone_angle:
            s.optional.ligand.cone_angle = cone_angle_schema.validate(s.optional.ligand.cone_angle)

        if s.optional.qd.optimize:
            qd_opt = s.optional.qd.optimize
            s.optional.qd.optimize = tblite_settings(qd_opt, qd_opt_schema.validate(qd_opt))
        if s.optional.qd.dissociate:
            bde_dict = tblite_settings(
                s.optional.qd.dissociate, bde_schema.validate(s.optional.qd.dissociate)
            )
            if (bde_dict["core_atom"] is None) is (bde_dict["core_index"] is None):
                raise ValueError("Exactly one of `core_atom` and `core_index` must be specified")
            elif bde_dict["lig_count"] is None:
                raise ValueError("`lig_count` must be specified")
            s.optional.qd.dissociate = parse_qmflows_keywords(bde_dict)
        if s.optional.qd.activation_strain:
            asa = s.optional.qd.activation_strain
            s.optional.qd.activation_strain = tblite_settings(asa, asa_schema.validate(asa))
        if s.optional.qd.multi_ligand:
            s.optional.qd.multi_ligand = multi_ligand_schema.validate(s.optional.qd.multi_ligand)
            _validate_multi_lig(s)
        if s.optional.qd.bulkiness:
            s.optional.qd.bulkiness = bulkiness_schema.validate(s.optional.qd.bulkiness)

        # Create forcefield Job Settings
        if s.optional.forcefield:
            s.optional.forcefield = validate_ff(s.optional.forcefield)
            update_ff_jobs(s)

        # Validate the input cores and ligands
        validate_mol(s.input_cores, 'input_cores', join(s.path, 'core'), not validate_only)
        validate_mol(s.input_ligands, 'input_ligands', join(s.path, 'ligand'), not validate_only)
        if 'input_qd' in s:
            validate_mol(s.input_qd, 'input_qd', join(s.path, 'qd'), not validate_only)

        # Create a dataCAT.Database instance
        if s.optional.database.get('db') is None and not validate_only:
            if DATA_CAT:
                db_path = s.optional.database.dirname
                s.optional.database.db = Database(path=db_path, **s.optional.database.mongodb)
            else:
                s.optional.database.db = None

        # Create RDKit molecules representing functional groups
        if s.optional.ligand.anchor is not None:
            func_groups = s.optional.ligand.anchor
        else:
            func_groups = s.optional.ligand.functional_groups
        del s.optional.ligand.functional_groups

        split = s.optional.ligand.split
        s.optional.ligand.anchor = parse_anchors(func_groups, split=split)
        s.optional.core.anchor = parse_anchors(s.optional.core.anchor, split=True, is_core=True)
