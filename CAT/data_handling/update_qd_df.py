"""A module for updating the index of quantum dot dataframes.

Index
-----
.. currentmodule:: CAT.data_handling.update_qd_df
.. autosummary::
    init_qd_import
    update_qd_df

API
---
.. autofunction:: init_qd_import
.. autofunction:: update_qd_df

"""

import pandas as pd

from scm.plams import Molecule

from ..workflows import WorkFlow, MOL, OPT, HDF5_INDEX

__all__ = ['init_qd_import', 'update_qd_df']


def init_qd_import(qd_df: pd.DataFrame) -> None:
    """Prepare a dataframe of imported quantum dots and store the quantum dots in the database.

    Storing them gives the quantum dots an HDF5 index, which later workflows
    (*e.g.* the activation strain analysis) need for storing their results.

    """
    update_qd_df(qd_df)
    workflow = WorkFlow.from_template(qd_df, name='qd_attach')
    workflow.to_db(qd_df)


def update_qd_df(qd_df: pd.DataFrame) -> None:
    """Update the index of **qd_df** with descriptors of the core(s) and ligand(s)."""
    # Construct the MultiIndex columns
    core = [_get_core_formula(mol) for mol in qd_df[MOL]]
    # There are no core dummy atoms to refer to; use the (unique) name of each quantum dot instead
    core_anchor = [mol.properties.name for mol in qd_df[MOL]]
    ligand_smiles = [mol.properties.pop('ligand_smiles') for mol in qd_df[MOL]]
    ligand_anchor = [mol.properties.pop('ligand_anchor') for mol in qd_df[MOL]]

    # Reassign the index
    idx_iter = zip(*(core, core_anchor, ligand_smiles, ligand_anchor))
    names = ['core', 'core anchor', 'ligand smiles', 'ligand anchor']
    qd_df.index = pd.MultiIndex.from_tuples(idx_iter, names=names)

    # Set the ('opt', '') column to False
    qd_df[OPT] = False
    qd_df[HDF5_INDEX] = -1


def _get_core_formula(mol: Molecule) -> str:
    """Construct a string representing the molecular formula of the core."""
    # Create a dictionary with atomic symbols and matching atom counts
    core = {}
    for i in mol.properties.indices:
        at = mol[i]
        if at.properties.get('anchor'):
            break  # The first ligand has been reached; abort and return
        symbol = at.symbol
        try:
            core[symbol] += 1
        except KeyError:
            core[symbol] = 1

    # Concatenate the dictionary into a single string
    return ''.join(f'{k}{v}' for k, v in sorted(core.items()))
