"""
nanoCAT.recipes.descriptors
===========================

A recipe for computing a table of ligand descriptors, *e.g.* for screening ligands.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    ligand_descriptors

API
---
.. autofunction:: ligand_descriptors

"""

from __future__ import annotations

import warnings
import functools
from itertools import chain
from typing import Any, TYPE_CHECKING
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scm.plams import Molecule

from CAT.workflows import MOL, V_BULK, BRANCH_DISTANCE, BRANCH_SIZE
from CAT.attachment.ligand_opt import optimize_ligand, allign_axis
from CAT.attachment.ligand_anchoring import find_substructure
from CAT.data_handling.anchor_parsing import parse_anchors

from nanoCAT.cone_angle import get_cone_angle
from nanoCAT.mol_bulk import get_V, get_lig_radius
from nanoCAT.bulk.mol_graph2 import get_branching_graph
from nanoCAT.recipes.bulk import read_smiles

if TYPE_CHECKING:
    from CAT.data_handling.anchor_parsing import AnchorTup

__all__ = ['ligand_descriptors']

_Row = tuple[tuple[str, str], list[Any]]


def ligand_descriptors(
    smiles_list: Iterable[str],
    anchor: None | str | Iterable[str] = None,
    *,
    split: bool = True,
    anchor_condition: None | Callable[[int], bool] = None,
    optimize: bool = True,
    cone_angle_dist: float | Sequence[float] = 0.0,
    diameter: None | float = 4.5,
    height_lim: None | float = 10.0,
    processes: int = 1,
    return_mol: bool = False,
) -> pd.DataFrame:
    r"""Compute the cone angle, bulkiness and branch distances of ligands from SMILES strings.

    The ligands are prepared as by ``init_cat``: the anchoring groups are identified
    (and split, *e.g.* carboxylic acids become carboxylates) and the ligands are
    optimized with RDKit UFF.
    The descriptors are computed as by the ``optional.ligand.cone_angle``,
    ``optional.ligand.branch_distance`` and ``optional.qd.bulkiness`` workflows,
    and the returned DataFrame has the same index and columns as the CAT databases.

    Examples
    --------
    .. code:: python

        >>> from nanoCAT.recipes import ligand_descriptors

        >>> smiles_list = ['CC(=O)O', 'CC(C)CC(=O)O', 'O=C(O)c1ccccc1']
        >>> df = ligand_descriptors(smiles_list)  # doctest: +SKIP
        >>> print(df.round(2))  # doctest: +SKIP
        index                    cone_angle V_bulk branch_distance branch_size
        sub index                  dist=0.0
        smiles            anchor
        CC(=O)[O-]        O3          82.94   0.25               0           7
        CC(C)CC(=O)[O-]   O6         105.89   6.79               0          16
        O=C([O-])c1ccccc1 O8         110.52   9.56             0-1        11-3

    Parameters
    ----------
    smiles_list : :class:`Iterable[str] <collections.abc.Iterable>`
        The SMILES strings of the ligands.
    anchor : :class:`str` or :class:`Iterable[str] <collections.abc.Iterable>`, optional
        One or more anchoring groups, in the format of ``optional.ligand.anchor``.
        If :data:`None`, use the default anchoring groups of CAT.
        A ligand with several anchoring groups yields one row per group.
    split : :class:`bool`
        Whether to remove a proton or counterion from the anchoring group
        (see ``optional.ligand.split``).
    anchor_condition : :class:`Callable[[int], bool] <collections.abc.Callable>`, optional
        If not :data:`None`, only accept ligands for which the number of identified
        anchoring groups satisfies this condition, *e.g.* ``lambda n: n == 1``.
        It must be picklable (*i.e.* not a lambda) if ``processes > 1``.
    optimize : :class:`bool`
        Whether to optimize the ligands (RDKit UFF) or only align them with the X-axis.
    cone_angle_dist : :class:`float` or :class:`Sequence[float] <collections.abc.Sequence>`
        One or more distances (Angstrom) of the anchor atom w.r.t. the surface,
        for the cone angle (see ``optional.ligand.cone_angle.distance``).
    diameter : :class:`float`, optional
        The lattice spacing :math:`d` (Angstrom) for :math:`V_{bulk}`,
        *i.e.* the average nearest-neighbor distance between the anchor atoms
        of the ligands (see ``optional.qd.bulkiness.d``).
        Set to :data:`None` to ignore the lattice spacing.
    height_lim : :class:`float`, optional
        The height cutoff :math:`h_{lim}` (Angstrom) for :math:`V_{bulk}`
        (see ``optional.qd.bulkiness.h_lim``).
        Set to :data:`None` to ignore the height cutoff.
    processes : :class:`int`
        The number of worker processes.
        This requires a version of CAT with the ``optional.processes`` option,
        and scripts using it must guard their entry point with ``if __name__ == "__main__":``.
    return_mol : :class:`bool`
        Whether to also return the (optimized) ligands, in the ``("mol", "")`` column.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame indexed by the ligands' canonical SMILES string and anchor atom, with:

        * ``("cone_angle", "dist=...")``: the cone angle (degrees), one column per distance.
        * ``("V_bulk", "")``: the bulkiness descriptor :math:`V_{bulk}`.
        * ``("branch_distance", "")`` and ``("branch_size", "")``: the distance of each branch
          to the anchor and the number of atoms in each branch, as dash-separated strings.

        Descriptors that cannot be computed for a ligand are set to ``NaN``
        (with a :exc:`RuntimeWarning`).

    """
    dist = np.asarray(cone_angle_dist, dtype=np.float64)
    if dist.ndim > 1:
        raise ValueError("`cone_angle_dist` should be a scalar or 1D sequence")
    dist_list = [float(dist)] if dist.ndim == 0 else [float(i) for i in dist]

    columns = [('cone_angle', f'dist={i}') for i in dist_list]
    columns += [V_BULK, BRANCH_DISTANCE, BRANCH_SIZE]
    if return_mol:
        columns.append(MOL)

    mol_list = read_smiles(smiles_list)
    anchors = parse_anchors(anchor, split=split)
    kwargs: dict[str, Any] = {
        'anchors': anchors, 'split': split, 'condition': anchor_condition, 'optimize': optimize,
        'dist_list': dist_list, 'diameter': diameter, 'height_lim': height_lim,
        'return_mol': return_mol,
    }
    if processes == 1:
        rows = list(chain.from_iterable(_get_rows(mol, **kwargs) for mol in mol_list))
    else:
        from CAT.parallel import get_processes, parallel_map, unpack_atom_refs
        func = functools.partial(_get_rows, pack_refs=True, **kwargs)
        n = get_processes(processes)
        rows = list(chain.from_iterable(parallel_map(func, [(m,) for m in mol_list], n)))
        if return_mol:
            for _, values in rows:
                ligand, refs = values[-1]
                unpack_atom_refs(ligand, refs)
                values[-1] = ligand

    if rows:
        index = pd.MultiIndex.from_tuples([i for i, _ in rows], names=['smiles', 'anchor'])
    else:
        index = pd.MultiIndex.from_arrays([[], []], names=['smiles', 'anchor'])
    df = pd.DataFrame(
        [values for _, values in rows],
        index=index,
        columns=pd.MultiIndex.from_tuples(columns, names=['index', 'sub index']),
    )
    return df[~df.index.duplicated(keep='first')]


def _get_rows(
    mol: Molecule,
    anchors: Sequence[AnchorTup],
    split: bool,
    condition: None | Callable[[int], bool],
    optimize: bool,
    dist_list: list[float],
    diameter: None | float,
    height_lim: None | float,
    return_mol: bool,
    pack_refs: bool = False,
) -> list[_Row]:
    """Return the index and descriptors of each anchoring group of **mol**."""
    ret = []
    for ligand in find_substructure(mol, anchors, split=split, condition=condition):
        values = _get_descriptors(ligand, optimize, dist_list, diameter, height_lim)
        if return_mol and pack_refs:
            # PLAMS does not pickle the atoms referred to by the ligand's properties
            from CAT.parallel import pack_atom_refs
            values.append((ligand, pack_atom_refs(ligand)))
        elif return_mol:
            values.append(ligand)
        ret.append(((ligand.properties.smiles, ligand.properties.anchor), values))
    return ret


def _get_descriptors(
    ligand: Molecule,
    optimize: bool,
    dist_list: list[float],
    diameter: None | float,
    height_lim: None | float,
) -> list[Any]:
    """Optimize **ligand** and return its cone angle(s), bulkiness and branch distances."""
    smiles = ligand.properties.smiles
    try:
        if optimize:
            optimize_ligand(ligand)
        else:
            allign_axis(ligand)
    except Exception as ex:
        _warn(f"Failed to optimize {smiles!r}", ex)
        return [np.nan] * (len(dist_list) + 3)

    ret: list[Any] = []
    try:
        anchor = ligand.atoms.index(ligand.properties.dummies)
        ret += [float(i) for i in np.atleast_1d(get_cone_angle(ligand, anchor, dist_list))]
    except Exception as ex:
        _warn(f"Failed to compute the cone angle of {smiles!r}", ex)
        ret += [np.nan] * len(dist_list)

    try:
        r, h = get_lig_radius(ligand)
        ret.append(float(get_V(r, h, diameter, None, h_lim=height_lim)))
    except Exception as ex:
        _warn(f"Failed to compute the bulkiness of {smiles!r}", ex)
        ret.append(np.nan)

    try:
        graph = get_branching_graph(ligand, ligand.properties.dummies)
    except Exception as ex:
        _warn(f"Failed to compute the branch distance of {smiles!r}", ex)
        ret += [np.nan, np.nan]
    else:
        ret.append("-".join(str(i) for i in graph.branch_distance))
        ret.append("-".join(str(i) for i in graph.branch_size))
    return ret


def _warn(msg: str, ex: Exception) -> None:
    warning = RuntimeWarning(msg)
    warning.__cause__ = ex
    warnings.warn(warning, stacklevel=3)
