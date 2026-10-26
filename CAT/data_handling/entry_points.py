#!/usr/bin/env python
"""Entry points for CAT.

Index
-----
.. currentmodule:: CAT.data_handling.entry_points
.. autosummary::
    extract_args
    get_input_template
    check_input
    main

API
---
.. autofunction:: extract_args
.. autofunction:: get_input_template
.. autofunction:: check_input
.. autofunction:: main

"""

import os
import shutil
import argparse
from os import getcwd
from os.path import join, exists
from importlib import resources
from typing import Any, Optional, List, Iterable, Tuple

import yaml

from scm.plams import Settings

from CAT import base
from CAT.logger import logger
from CAT.utils import log_traceback_locals
from CAT.data_handling.mol_import import read_mol
from CAT.data_handling.validate_input import validate_input

__all__ = ['main', 'check_input', 'get_input_template', 'extract_args']

#: Jobs that need the Amsterdam Modeling Suite
_AMS_JOBS = frozenset({'ADFJob', 'AMSJob', 'CRSJob', 'DFTBJob', 'BANDJob', 'ReaxFFJob'})

#: Names of the CP2K executables
_CP2K_EXECUTABLES = ('cp2k.popt', 'cp2k.psmp', 'cp2k.ssmp', 'cp2k')


def extract_args(args: Any) -> Settings:
    """Read the input file of the parsed command-line arguments **args** into a Settings object."""
    input_file = args.YAML[0] if isinstance(args.YAML, list) else args.YAML
    if exists(input_file):
        pass
    elif exists(join(getcwd(), input_file)):
        input_file = join(getcwd(), input_file)
    else:
        input_file2 = join(getcwd(), input_file)
        raise FileNotFoundError(f'No file found at {input_file} or {input_file2}')

    with open(input_file, 'r') as file:
        return Settings(yaml.load(file, Loader=yaml.FullLoader))


def get_input_template() -> str:
    """Return a commented template of a CAT input file.

    Examples
    --------
    .. code:: python

        >>> import yaml
        >>> from CAT.data_handling.entry_points import get_input_template

        >>> sorted(yaml.safe_load(get_input_template()))
        ['input_cores', 'input_ligands', 'optional', 'path']

    """
    ref = resources.files('CAT') / 'data' / 'templates' / 'input_template.yaml'
    return ref.read_text(encoding='utf-8')


def _job_names(block: Any, keys: Iterable[str] = ('job1', 'job2')) -> List[str]:
    """Return the names of the PLAMS jobs in a validated workflow block."""
    if not isinstance(block, dict):
        return []
    ret = []
    for key in keys:
        job = block.get(key)
        if job is not None:
            ret.append(getattr(job, '__name__', str(job)))
    return ret


def _get_workflows(s: Settings) -> List[Tuple[str, List[str], bool]]:
    """Return the name, jobs and whether Nano-CAT is required of every enabled workflow."""
    lig = s.optional.ligand
    qd = s.optional.qd
    #: Workflows that default to RDKit's UFF if no jobs are specified
    uff = {'Ligand geometry optimization', 'Activation strain analysis'}
    candidates = [
        ('Ligand geometry optimization', lig.optimize, False),
        ('Ligand COSMO-RS calculations', lig.get('crs'), True),
        ('Ligand conceptual DFT calculations', lig.cdft, True),
        ('Ligand cone angles', lig.cone_angle, True),
        ('Ligand branch distances', lig.branch_distance, True),
        ('Ligand forcefield assignment (MATCH)', s.optional.forcefield, True),
        ('Quantum dot construction', qd.construct_qd, False),
        ('Quantum dot optimization', qd.optimize, False),
        ('Multi-ligand attachment', qd.multi_ligand, False),
        ('Ligand bulkiness', qd.bulkiness, True),
        ('Activation strain analysis', qd.activation_strain, True),
        ('Ligand dissociation energies', qd.dissociate, True),
    ]
    ret = []
    for name, block, nano_cat in candidates:
        if not block:
            continue
        jobs = _job_names(block)
        if name == 'Ligand COSMO-RS calculations' and not jobs:
            jobs = ['AMSJob', 'CRSJob']
        elif name in uff and not jobs:
            jobs = ['RDKit UFF']
        ret.append((name, jobs, nano_cat))
    return ret


def check_input(s: Settings, read_molecules: bool = False) -> str:
    """Validate the CAT input **s** without running anything or creating any files.

    Parameters
    ----------
    s : |plams.Settings|
        The CAT input settings; they are updated inplace.
    read_molecules : bool
        Whether to also read all input molecules and list the workflows that would run.

    Returns
    -------
    str
        A short report.

    """
    validate_input(s, validate_only=True)
    lines = ['The input is valid.']
    if not read_molecules:
        return '\n'.join(lines)

    counts = {}
    for key in ('input_cores', 'input_ligands', 'input_qd'):
        if s.get(key):
            mol_list = read_mol(s[key])
            counts[key] = 0 if mol_list is None else len(mol_list)

    if 'input_qd' in counts:
        lines.append(f"Read {counts['input_qd']} quantum dot(s).")
    else:
        n_core = counts.get('input_cores', 0)
        n_lig = counts.get('input_ligands', 0)
        lines.append(
            f"Read {n_core} core(s) and {n_lig} ligand(s): {n_core * n_lig} quantum dot(s) "
            "(more if a ligand has several anchors)."
        )

    workflows = _get_workflows(s)
    lines.append('Workflows that would run:' if workflows else 'No workflows are enabled.')
    has_cp2k = any(shutil.which(exe) for exe in _CP2K_EXECUTABLES)
    for name, jobs, nano_cat in workflows:
        notes = []
        if jobs:
            notes.append(', '.join(jobs))
        if nano_cat and base.NANO_CAT is not None:
            notes.append('needs Nano-CAT, which could not be imported')
        if _AMS_JOBS.intersection(jobs) and not os.environ.get('AMSBIN'):
            notes.append('needs AMS, but AMSBIN is not set')
        if 'Cp2kJob' in jobs and not has_cp2k:
            notes.append('needs CP2K, which is not on the PATH')
        suffix = f" ({'; '.join(notes)})" if notes else ''
        lines.append(f'  - {name}{suffix}')
    return '\n'.join(lines)


def _log_summary(s: Settings, ret: Optional[tuple]) -> None:
    """Log the number of molecules produced by :func:`CAT.base.prep`."""
    if ret is None:
        return
    qd_df, core_df, ligand_df = ret
    parts = []
    for name, df, key in [('ligand(s)', ligand_df, 'ligand'), ('core(s)', core_df, 'core'),
                          ('quantum dot(s)', qd_df, 'qd')]:
        if df is not None:
            parts.append(f'{len(df)} {name} in {s.optional[key].dirname!r}')
    if parts:
        logger.info('Results: ' + ', '.join(parts))


def main(args: Optional[List[str]] = None) -> None:
    """Launch CAT via the command line."""
    parser = argparse.ArgumentParser(
        prog='init_cat',
        usage='init_cat [--validate | --dry-run | --template] [input.yaml]',
        description='Run the Compound Attachment Tool (CAT) with the settings in input.yaml.',
    )

    parser.add_argument(
        'YAML', nargs='?', type=str, metavar='input.yaml',
        help='A .yaml file with the settings for CAT',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--validate', action='store_true',
        help='Only check the input file; do not run anything or create any files',
    )
    group.add_argument(
        '--dry-run', action='store_true',
        help=('Check the input file, read all input molecules and list the workflows '
              'that would run, without running them or creating any files'),
    )
    group.add_argument(
        '--template', action='store_true',
        help='Print a commented template input file (e.g. `init_cat --template > input.yaml`)',
    )

    try:
        parsed = parser.parse_args(args)
        if parsed.template:
            print(get_input_template(), end='')
            return
        elif parsed.YAML is None:
            parser.error('the following arguments are required: input.yaml')

        s = extract_args(parsed)
        if parsed.validate or parsed.dry_run:
            print(check_input(s, read_molecules=parsed.dry_run))
            return

        _log_summary(s, base.prep(s, return_mol=True))
    except Exception:
        log_traceback_locals(logger)
        raise
