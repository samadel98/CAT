"""A module related to the importing of molecules.

Index
-----
.. currentmodule:: CAT.data_handling.mol_import
.. autosummary::
    read_mol
    read_mol_xyz
    read_mol_pdb
    read_mol_mol
    read_mol_smiles
    read_mol_plams
    read_mol_rdkit
    read_mol_folder
    read_mol_txt
    get_charge_dict
    set_mol_prop
    set_atom_prop
    set_qd
    find_ligands
    print_exception

API
---
.. autofunction:: read_mol
.. autofunction:: read_mol_xyz
.. autofunction:: read_mol_pdb
.. autofunction:: read_mol_mol
.. autofunction:: read_mol_smiles
.. autofunction:: read_mol_plams
.. autofunction:: read_mol_rdkit
.. autofunction:: read_mol_folder
.. autofunction:: read_mol_txt
.. autofunction:: get_charge_dict
.. autofunction:: set_mol_prop
.. autofunction:: set_atom_prop
.. autofunction:: set_qd
.. autofunction:: find_ligands
.. autofunction:: print_exception

"""

import os
import itertools
from types import MappingProxyType
from string import ascii_letters
from typing import (
    Dict, Iterable, List, Sequence, Optional, Mapping, Callable, Tuple, overload
)

import numpy as np
from scipy.spatial import cKDTree

from scm.plams import (Molecule, Atom, Settings, MoleculeError)
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit import Chem, RDLogger

from ..utils import get_formula
from ..logger import logger
from ..mol_utils import rdkit_props
from .._compat import get_conformation, readpdb
from ..data_handling.validate_mol import validate_mol

__all__ = ['read_mol', 'set_mol_prop']

# Supress the rdkit logger
_logger = RDLogger.logger()
_logger.setLevel(RDLogger.CRITICAL)


@overload
def read_mol(input_mol: Iterable[Settings]) -> List[Molecule]:
    ...
@overload  # noqa: E302
def read_mol(input_mol: None) -> None:
    ...
def read_mol(input_mol):  # noqa: E302
    """Checks the filetypes of the input molecules.

    Sets the molecules' properties and returns a list of plams molecules.

    Parameters
    ----------
    input_mol : |list|_ [|Settings|_]
        An iterable consisting of dictionaries with input settings per mol.

    Returns
    -------
    |plams.Molecule|_
        A list of plams Molecules.

    """
    if input_mol is None:
        return None

    # Create a list of PLAMS molecules
    mol_list = []
    append = mol_list.append
    extend = mol_list.extend

    for mol_dict in input_mol:
        try:
            read_mol = EXTENSION_MAPPING[mol_dict.type]
        except KeyError as ex:
            logger.error(f'{ex.__class__.__name__}: {ex}')
            continue

        mol = read_mol(mol_dict)
        if not mol:  # Failed to import any molecules
            continue

        if isinstance(mol, list):  # if mol is a list of molecules
            extend(mol)

        elif mol_dict.is_qd:  # A quantum dot molecule
            set_qd(mol, mol_dict)
            append(mol)

        else:  # A core or ligand molecule
            if mol_dict.guess_bonds:
                mol.guess_bonds()
            set_mol_prop(mol, mol_dict)
            append(mol)

    return mol_list


def read_mol_xyz(mol_dict: Settings) -> Optional[Molecule]:
    """Read an .xyz file."""
    try:
        mol = Molecule(mol_dict.mol, inputformat='xyz')
        if mol_dict.guess_bonds and not mol_dict.is_qd:
            mol.guess_bonds()
        if mol_dict.canonicalize:
            canonicalize_mol(mol)
        return mol
    except Exception as ex:
        print_exception('read_mol_xyz', mol_dict.name, ex)


def read_mol_pdb(mol_dict: Settings) -> Optional[Molecule]:
    """Read a .pdb file."""
    try:
        mol = readpdb(mol_dict.mol)
        if mol_dict.guess_bonds and not mol_dict.is_qd:
            mol.guess_bonds()
        if mol_dict.canonicalize:
            canonicalize_mol(mol)
        return mol
    except Exception as ex:
        print_exception('read_mol_pdb', mol_dict.name, ex)


def read_mol_mol(mol_dict: Settings) -> Optional[Molecule]:
    """Read a .mol file."""
    try:
        mol = molkit.from_rdmol(Chem.MolFromMolFile(mol_dict.mol, removeHs=False))
        if mol_dict.guess_bonds and not mol_dict.is_qd:
            mol.guess_bonds()
        if mol_dict.canonicalize:
            canonicalize_mol(mol)
        return mol
    except Exception as ex:
        print_exception('read_mol_mol', mol_dict.name, ex)


def read_mol_smiles(mol_dict: Settings) -> Optional[Molecule]:
    """Read a SMILES string."""
    try:
        mol = _from_smiles(mol_dict.mol)
        mol.properties.smiles = Chem.CanonSmiles(mol_dict.mol)

        if mol_dict.get('indices'):
            for i in mol_dict.indices:
                mol[i].properties.anchor = True

        canonicalize_mol(mol)
        if mol_dict.get('indices'):
            mol_dict.indices = tuple(i for i, at in enumerate(mol, 1) if
                                     at.properties.pop('anchor', False))

        if mol_dict.guess_bonds and not mol_dict.is_qd:
            mol.guess_bonds()
        return mol
    except Exception as ex:
        print_exception('read_mol_smiles', mol_dict.name, ex)


SMILES_PARAMS = Chem.SmilesParserParams()
SMILES_PARAMS.removeHs = False


def _from_smiles(smiles: str) -> Molecule:
    _rdmol = Chem.MolFromSmiles(smiles, params=SMILES_PARAMS)
    rdmol = Chem.AddHs(_rdmol)
    rdmol.SetProp('smiles', smiles)
    return get_conformation(rdmol)


def read_mol_plams(mol_dict: Settings) -> Optional[Molecule]:
    """Read a PLAMS molecule."""
    try:
        mol = mol_dict.mol
        if mol_dict.guess_bonds and not mol_dict.is_qd:
            mol.guess_bonds()
        if mol_dict.canonicalize:
            canonicalize_mol(mol)
        return mol
    except Exception as ex:
        print_exception('read_mol_plams', mol_dict.name, ex)


def read_mol_rdkit(mol_dict: Settings) -> Optional[Molecule]:
    """Read a RDKit molecule."""
    try:
        mol = molkit.from_rdmol(mol_dict.mol)
        if mol_dict.guess_bonds and not mol_dict.is_qd:
            mol.guess_bonds()
        if mol_dict.canonicalize:
            canonicalize_mol(mol)
        return mol
    except Exception as ex:
        print_exception('read_mol_rdkit', mol_dict.name, ex)


def read_mol_folder(mol_dict: Settings) -> Optional[List[Molecule]]:
    """Read all files (.xyz, .pdb, .mol, .txt or further subfolders) within a folder."""
    try:
        mol_type = 'input_cores' if mol_dict.is_core else 'input_ligands'

        _file_list = os.listdir(mol_dict.mol)
        optional_dict = Settings({k: v for k, v in mol_dict.items() if k not in ('mol', 'path')})
        file_list = [{i: optional_dict} for i in _file_list]

        validate_mol(file_list, mol_type, mol_dict.path)
        return read_mol(file_list)
    except Exception as ex:
        print_exception('read_mol_folder', mol_dict.name, ex)


def read_mol_txt(mol_dict: Settings) -> Optional[List[Molecule]]:
    """Read a plain text file containing one or more SMILES strings."""
    try:
        row = 0 if 'row' not in mol_dict else mol_dict.row
        column = 0 if 'column' not in mol_dict else mol_dict.column
        mol_type = 'input_cores' if mol_dict.is_core else 'input_ligands'

        with open(mol_dict.mol, 'r') as f:
            iterator = itertools.islice(f, row, None)
            _file_list = [i.rstrip('\n').split()[column] for i in iterator if i]
        optional_dict = Settings({k: v for k, v in mol_dict.items() if k not in ('mol', 'path')})
        file_list = [{i: optional_dict} for i in _file_list]

        validate_mol(file_list, mol_type, mol_dict.path)
        return read_mol(file_list)
    except Exception as ex:
        print_exception('read_mol_txt', mol_dict.name, ex)


EXTENSION_MAPPING: Mapping[str, Callable[[Settings], Optional[Molecule]]] = MappingProxyType({
    'xyz': read_mol_xyz,
    'pdb': read_mol_pdb,
    'mol': read_mol_mol,
    'smiles': read_mol_smiles,
    'folder': read_mol_folder,
    'txt': read_mol_txt,
    'plams_mol': read_mol_plams,
    'rdmol': read_mol_rdkit
})


def canonicalize_mol(mol: Molecule, inplace: bool = True) -> Optional[Molecule]:
    """Take a PLAMS molecule and sort its atoms based on their canonical rank.

    .. _rdkit.Chem.CanonicalRankAtoms: https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.CanonicalRankAtoms

    Examples
    --------
    .. code:: python

        >>> from scm.plams import Molecule, from_smiles

        # Methane
        >>> mol: Molecule = from_smiles('C')
        >>> print(mol)  # doctest: +SKIP
        Atoms:
            1         H      0.640510      0.640510     -0.640510
            2         H      0.640510     -0.640510      0.640510
            3         C      0.000000      0.000000      0.000000
            4         H     -0.640510      0.640510      0.640510
            5         H     -0.640510     -0.640510     -0.640510

        >>> canonicalize_mol(mol)
        >>> print(mol)  # doctest: +SKIP
        Atoms:
            1         C      0.000000      0.000000      0.000000
            2         H     -0.640510     -0.640510     -0.640510
            3         H     -0.640510      0.640510      0.640510
            4         H      0.640510     -0.640510      0.640510
            5         H      0.640510      0.640510     -0.640510

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    inplace : bool
        If ``True``, perform an inplace update of **mol** rather than returning
        a new :class:`Molecule` instance.

    Returns
    -------
    |plams.Molecule|_
        Optional: if ``inplace=False``, return a copy of **mol** with its atoms sorted by their
        canonical rank.

    See Also
    --------
    * rdkit.Chem.CanonicalRankAtoms_: Returns the canonical atom ranking for each atom of a
      molecule fragment.

    """  # noqa
    rdmol = molkit.to_rdmol(mol, properties=False)
    idx_collection = Chem.CanonicalRankAtoms(rdmol)

    # Reverse sort Molecule.atoms by the atomic indices in idx_collection
    if inplace:
        mol.atoms = [at for _, at in sorted(zip(idx_collection, mol.atoms), reverse=True)]
        return
    else:
        ret = mol.copy()
        ret.atoms = [at for _, at in sorted(zip(idx_collection, ret.atoms), reverse=True)]
        return ret


def get_charge_dict() -> Dict[str, int]:
    """Create a dictionary of elements and their formal atomic charge."""
    elements = [
        (1, ['Li', 'Na', 'K', 'Rb', 'Cs']),         # Group 1: Alkali metals
        (2, ['Be', 'Mg', 'Ca', 'Sr', 'Ba']),        # Group 2: Alkaline earth metals
        (-3, ['N', 'P', 'As', 'Sb', 'Bi']),         # Group 15: pnictogens
        (-2, ['O', 'S', 'Se', 'Te', 'Po']),         # Group 16: Chalcogens
        (-1, ['H', 'F', 'Cl', 'Br', 'I', 'At']),    # Group 17: Halogens
        (2, ['Cd', 'Pb']),                          # Misc 2+
        (3, ['In']),                                # Misc 3+
    ]
    return {at: q for q, at_list in elements for at in at_list}


#: A dictionary with default atomic charges
CHARGE_MAPPING: Mapping[str, int] = MappingProxyType(get_charge_dict())


def set_mol_prop(mol: Molecule, mol_dict: Settings) -> None:
    """Set molecular and atomic properties."""
    if mol_dict.is_core:
        residue_name = 'COR'
        mol.properties.name = get_formula(mol)
    else:
        residue_name = 'LIG'
        mol.properties.name = mol_dict.name

    mol.properties.dummies = mol_dict.get("indices")
    mol.properties.path = mol_dict.path
    mol.properties.job_path = []

    # Prepare a generator of letters for pdb_info.Name
    alphabet = itertools.combinations(ascii_letters, 2)

    # Set the atomic properties
    for atom, i in zip(mol, itertools.cycle(alphabet)):
        set_atom_prop(atom, i, residue_name)

    if not mol.properties.smiles:
        mol.properties.smiles = Chem.MolToSmiles(
            Chem.RemoveHs(molkit.to_rdmol(mol, properties=False)), canonical=True
        )


def set_atom_prop(atom: Atom, at_id: Sequence[str], residue_name: str) -> None:
    """Set atomic properties."""
    symbol = '{:4}'.format(atom.symbol + ''.join(at_id))

    # Add a number of properties to atom
    rdkit_props(atom).pdb_info.ResidueName = residue_name
    rdkit_props(atom).pdb_info.Occupancy = 1.0
    rdkit_props(atom).pdb_info.TempFactor = 0.0
    rdkit_props(atom).pdb_info.ResidueNumber = 1
    rdkit_props(atom).pdb_info.Name = symbol
    rdkit_props(atom).pdb_info.ChainId = 'A'

    # Changes hydrogen and carbon from heteroatom to atom
    if atom.symbol in ('H', 'C'):
        rdkit_props(atom).pdb_info.IsHeteroAtom = False
    else:
        rdkit_props(atom).pdb_info.IsHeteroAtom = True

    # Sets the formal atomic charge
    if rdkit_props(atom).charge:
        return

    # Default to a charge of 0 if no charge is available for that specific element
    if atom.symbol not in CHARGE_MAPPING:
        rdkit_props(atom).charge = 0
        return

    # Update the charge of non-hypervalent atoms
    total_bonds = int(sum([bond.order for bond in atom.bonds]))
    default_charge = CHARGE_MAPPING[atom.symbol]
    abs_charge = abs(default_charge)
    sign = -1 * int(default_charge / abs_charge)

    # Take the default charge and correct for the number (and order) of bonds
    rdkit_props(atom).charge = default_charge + sign * total_bonds
    if total_bonds <= abs_charge:
        return

    # Update formal atomic charges for hypervalent atoms
    if total_bonds is 2 + abs_charge:
        rdkit_props(atom).charge -= 2 * sign
    elif total_bonds is 4 + abs_charge:
        rdkit_props(atom).charge -= 4 * sign
    elif total_bonds >= 6 + abs_charge:
        rdkit_props(atom).charge -= 6 * sign
    return


def set_qd(qd: Molecule, mol_dict: Settings) -> Molecule:
    """Update quantum dots imported by :func:`.read_mol`.

    The ligands are identified by matching the connectivity of ``mol_dict.ligand_smiles``,
    after guessing all bonds if the file did not provide any (or ``guess_bonds = True``).
    The atoms are then reordered (core first, followed by one ligand at a time),
    the bonds and charges of the ligands are taken from the SMILES string
    and all properties used by the CAT and Nano-CAT workflows are set.

    """
    if not (mol_dict.get('ligand_smiles') and mol_dict.get('ligand_anchor')):
        raise ValueError(f"{mol_dict.name!r}: imported quantum dots require "
                         "the 'ligand_smiles' and 'ligand_anchor' arguments")

    # Create the ligand (with explicit hydrogens) and find its anchor atom
    ligand = Chem.AddHs(Chem.MolFromSmiles(mol_dict.ligand_smiles))
    anchor_match = ligand.GetSubstructMatch(Chem.MolFromSmarts(mol_dict.ligand_anchor))
    if not anchor_match:
        raise MoleculeError(f"{mol_dict.name!r}: ligand_anchor {mol_dict.ligand_anchor!r} "
                            f"is not part of ligand_smiles {mol_dict.ligand_smiles!r}")
    _anchor_idx = anchor_match[0]

    # Guess the connectivity if the file did not provide any
    guess_bonds = bool(mol_dict.guess_bonds or not qd.bonds)
    if guess_bonds:
        qd.guess_bonds()

    # Find the ligands; all remaining atoms belong to the core
    lig_idx = find_ligands(qd, ligand, _anchor_idx)
    if not len(lig_idx):
        raise MoleculeError(f"{mol_dict.name!r}: no ligands matching "
                            f"{mol_dict.ligand_smiles!r} were found")
    is_lig = np.zeros(len(qd), dtype=bool)
    is_lig[lig_idx.ravel()] = True
    core_idx = np.flatnonzero(~is_lig)

    # Reorder all atoms: core atoms first followed by ligands
    atoms = qd.atoms
    qd.atoms = [atoms[i] for i in core_idx] + [atoms[i] for i in lig_idx.ravel()]

    # Replace the (guessed) ligand bonds with those of the ligand
    if guess_bonds:
        qd.delete_all_bonds()
    lig_atoms = [[atoms[i] for i in idx] for idx in lig_idx]
    for lig in lig_atoms:
        lig_set = set(lig)
        intra_bonds = {id(b): b for at in lig for b in at.bonds if b.other_end(at) in lig_set}
        for bond in intra_bonds.values():
            qd.delete_bond(bond)
        for bond in ligand.GetBonds():
            at1 = lig[bond.GetBeginAtomIdx()]
            at2 = lig[bond.GetEndAtomIdx()]
            qd.add_bond(at1, at2, order=bond.GetBondTypeAsDouble())

    # Set the formal charges of the ligand atoms
    for lig in lig_atoms:
        for at, rdatom in zip(lig, ligand.GetAtoms()):
            rdkit_props(at).charge = rdatom.GetFormalCharge()

    # Set the pdb_info (and the charges of the core atoms)
    alphabet = itertools.cycle(itertools.combinations(ascii_letters, 2))
    for at, at_id in zip(qd.atoms[:len(core_idx)], alphabet):
        set_atom_prop(at, at_id, 'COR')
    for at, at_id in zip(qd.atoms[len(core_idx):], alphabet):
        set_atom_prop(at, at_id, 'LIG')

    # Construct a list with the indices of all ligand anchor atoms
    core_idx_max = 1 + len(core_idx)
    start = core_idx_max + _anchor_idx
    stop = core_idx_max + lig_idx.size
    step = ligand.GetNumAtoms()
    anchor_idx = list(range(start, stop, step))

    # Update the properties of **qd**
    for i in anchor_idx:
        qd[i].properties.anchor = True
    qd.properties.indices = list(range(1, core_idx_max)) + anchor_idx
    qd.properties.job_path = []
    qd.properties.name = mol_dict.name
    qd.properties.path = mol_dict.path
    qd.properties.ligand_smiles = Chem.CanonSmiles(mol_dict.ligand_smiles)
    i = 1 + _anchor_idx
    qd.properties.ligand_anchor = f'{ligand.GetAtomWithIdx(_anchor_idx).GetSymbol()}{i}'

    # Update the pdb_info of all atoms
    for i, at in enumerate(qd, 1):
        rdkit_props(at).pdb_info.SerialNumber = i
        if i < core_idx_max:  # A core atom
            rdkit_props(at).pdb_info.ResidueNumber = 1
        else:  # A ligand atom
            rdkit_props(at).pdb_info.ResidueNumber = 2 + int((i - core_idx_max) / step)
    return qd


def find_ligands(qd: Molecule, ligand: Chem.Mol, anchor: int = 0) -> np.ndarray:
    """Find all ligands in **qd** based on the connectivity of **ligand**.

    Heavy atoms are matched on their element and number of hydrogens, and bonds on their
    presence only, so neither formal charges nor (guessed) bond orders have to agree.
    Symmetry-equivalent atoms (*e.g.* the two oxygens of a carboxylate) are assigned such that
    the **anchor** atom is the one closest to the remaining (core) atoms,
    and, if still undecided, such that higher-order bonds are the shorter ones.

    Parameters
    ----------
    qd : |plams.Molecule|_
        A quantum dot (or any other molecule) with bonds.
    ligand : :class:`rdkit.Chem.Mol`
        The ligand, including explicit hydrogens.
    anchor : int
        The 0-based index of the anchor atom in **ligand**.

    Returns
    -------
    :class:`numpy.ndarray[int] <numpy.ndarray>`, shape :math:`(n_{lig}, n_{atom})`
        The 0-based indices of the atoms in **qd** belonging to each ligand,
        ordered like the atoms in **ligand**. Overlapping matches are discarded.

    """
    n = ligand.GetNumAtoms()
    atom_idx = {id(at): i for i, at in enumerate(qd.atoms)}
    bonds = [(atom_idx[id(b.atom1)], atom_idx[id(b.atom2)]) for b in qd.bonds]
    neighbors: List[List[int]] = [[] for _ in range(len(qd))]
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    xyz = np.array([at.coords for at in qd.atoms], dtype=np.float64)
    mapper = _LigandMapper(ligand, anchor, xyz, None, neighbors)
    if not mapper.heavy:
        return np.zeros((0, n), dtype=np.intp)

    # Match the heavy atoms only, with their number of hydrogens encoded as isotope.
    # Matching explicit hydrogens would enumerate all their permutations, which, before
    # RDKit 2022.09, also count towards `maxMatches` (so that most ligands were missed)
    atnums = [at.atnum for at in qd.atoms]
    h_neighbors = [sorted(j for j in nb if atnums[j] == 1) for nb in neighbors]
    heavy_qd = [i for i, atnum in enumerate(atnums) if atnum != 1]
    heavy_pos = {i: k for k, i in enumerate(heavy_qd)}
    target = _skeleton(
        (atnums[i] for i in heavy_qd),
        ((heavy_pos[i], heavy_pos[j]) for i, j in bonds if i in heavy_pos and j in heavy_pos),
        (1 + len(h_neighbors[i]) for i in heavy_qd),
    )

    # Find all non-overlapping ligands
    max_matches = len(mapper.perms) * max(1, len(heavy_qd) // len(mapper.heavy))
    matches: List[List[int]] = []
    used: set = set()
    for heavy_match in target.GetSubstructMatches(
        mapper.heavy_query, uniquify=True, maxMatches=max_matches
    ):
        mapping = {i: heavy_qd[k] for i, k in zip(mapper.heavy, heavy_match)}
        for i, h_list in mapper.hydrogens.items():
            mapping.update(zip(h_list, h_neighbors[mapping[i]]))
        match = [mapping[i] for i in range(n)]
        if used.isdisjoint(match):
            matches.append(match)
            used.update(match)
    if not matches:
        return np.zeros((0, n), dtype=np.intp)

    # Assign symmetry-equivalent atoms based on the geometry
    core_mask = np.ones(len(qd), dtype=bool)
    core_mask[list(used)] = False
    mapper.core_tree = cKDTree(xyz[core_mask]) if core_mask.any() else None
    return np.array([mapper(m) for m in matches], dtype=np.intp)


class _LigandMapper:
    """Pick the geometrically most sensible atom mapping of a ligand (see :func:`find_ligands`)."""

    def __init__(self, ligand: Chem.Mol, anchor: int, xyz: np.ndarray,
                 core_tree: "None | cKDTree", neighbors: List[List[int]]) -> None:
        heavy = [at.GetIdx() for at in ligand.GetAtoms() if at.GetAtomicNum() != 1]
        heavy_pos = {i: k for k, i in enumerate(heavy)}
        heavy_bonds = [b for b in ligand.GetBonds()
                       if b.GetBeginAtomIdx() in heavy_pos and b.GetEndAtomIdx() in heavy_pos]
        self.heavy = heavy
        self.h_idx = [at.GetIdx() for at in ligand.GetAtoms() if at.GetAtomicNum() == 1]
        self.hydrogens = {
            i: [nb.GetIdx() for nb in ligand.GetAtomWithIdx(i).GetNeighbors()
                if nb.GetAtomicNum() == 1]
            for i in heavy
        }

        #: The heavy atoms, with their number of hydrogens (plus one) as isotope
        self.heavy_query = _skeleton(
            (ligand.GetAtomWithIdx(i).GetAtomicNum() for i in heavy),
            ((heavy_pos[b.GetBeginAtomIdx()], heavy_pos[b.GetEndAtomIdx()]) for b in heavy_bonds),
            (1 + len(self.hydrogens[i]) for i in heavy),
        )

        #: All permutations of the heavy atoms that preserve their connectivity
        self.perms = [
            [heavy[k] for k in perm] for perm in
            self.heavy_query.GetSubstructMatches(
                self.heavy_query, uniquify=False, maxMatches=10000
            )
        ]
        self.bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
                      for b in heavy_bonds]
        self.n = ligand.GetNumAtoms()
        self.anchor = anchor
        self.xyz = xyz
        self.core_tree = core_tree
        self.neighbors = neighbors

    def __call__(self, match: Sequence[int]) -> List[int]:
        """Return the best mapping of the ligand atoms onto the quantum dot atoms in **match**."""
        if len(self.perms) <= 1:
            return list(match)

        h_atoms = {match[i] for i in self.h_idx}
        best: "None | Tuple[float, float]" = None
        ret = list(match)
        for perm in self.perms:
            # Map ligand atom ``heavy[k]`` onto the quantum dot atom of ligand atom ``perm[k]``
            mapping = {i: match[j] for i, j in zip(self.heavy, perm)}
            for i, h_list in self.hydrogens.items():
                h_target = [k for k in self.neighbors[mapping[i]] if k in h_atoms]
                if len(h_target) != len(h_list):
                    break  # The hydrogens do not fit this permutation
                mapping.update(zip(h_list, h_target))
            else:
                if self.core_tree is not None:
                    dist = float(self.core_tree.query(self.xyz[mapping[self.anchor]])[0])
                else:
                    dist = 0.0
                length = sum(
                    order * float(np.linalg.norm(self.xyz[mapping[i]] - self.xyz[mapping[j]]))
                    for i, j, order in self.bonds
                )
                score = (round(dist, 1), length)
                if best is None or score < best:
                    best = score
                    ret = [mapping[i] for i in range(self.n)]
        return ret


def _skeleton(atnums: Iterable[int], bonds: Iterable[Sequence[int]],
              isotopes: Optional[Iterable[int]] = None) -> Chem.Mol:
    """Construct an uncharged molecule with only single bonds from atomic numbers and bonds.

    Non-zero **isotopes** have to be equal for two atoms to match in a substructure search.

    """
    mol = Chem.RWMol()
    for atnum, isotope in zip(atnums, itertools.repeat(0) if isotopes is None else isotopes):
        atom = Chem.Atom(atnum)
        atom.SetNoImplicit(True)
        atom.SetIsotope(isotope)
        mol.AddAtom(atom)
    for i, j in bonds:
        mol.AddBond(i, j, Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol.GetMol()


def print_exception(func_name: str, mol_name: str, ex: Exception) -> None:
    """Manages the printing of exceptions upon failing to import a molecule."""
    ex_name = ex.__class__.__name__
    err = f'CAT.{func_name}() failed to parse {repr(mol_name)}; Caught exception: {ex_name}'
    logger.error(err)
