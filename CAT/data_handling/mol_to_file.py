"""A module designed for exporting PLAMS :class:`Molecule` instances to common file formats.

Index
-----
.. currentmodule:: CAT.data_handling.mol_to_file
.. autosummary::
    mol_to_file
    write_pdb
    shorten_filename

API
---
.. autofunction:: mol_to_file
.. autofunction:: write_pdb
.. autofunction:: shorten_filename

"""

import os
import errno
import hashlib
from types import MappingProxyType
from collections.abc import Iterable, Container, Callable
from os.path import join, isdir, isfile, exists

from rdkit import Chem, Geometry
from scm.plams import Molecule
from scm.plams.interfaces.molecule.rdkit import set_PDBresidueInfo

from ..logger import logger
from ..mol_utils import rdkit_props

__all__ = ['mol_to_file', 'write_pdb', 'shorten_filename']

MolExportFunc = Callable[[Molecule, str | bytes | os.PathLike], None]


def write_pdb(mol: Molecule, filename: str | bytes | os.PathLike) -> None:
    """Write **mol** to a .pdb file.

    Produces the same file as PLAMS' :func:`~scm.plams.interfaces.molecule.rdkit.writepdb`,
    but only converts the atomic properties that a .pdb file holds (charges and residue info)
    rather than pickling all of them, which makes it several times faster for quantum dots.

    """
    rdmol = Chem.RWMol()
    atom_idx = {}
    for i, at in enumerate(mol.atoms):
        rd_atom = Chem.Atom(int(at.atnum))
        props = rdkit_props(at)
        if 'charge' in props:
            rd_atom.SetFormalCharge(int(props.charge))
        if 'pdb_info' in props:
            set_PDBresidueInfo(rd_atom, props.pdb_info)
        rdmol.AddAtom(rd_atom)
        atom_idx[id(at)] = i

    for bond in mol.bonds:
        order = bond.order
        bond_type = Chem.BondType.AROMATIC if 1.4 < order < 1.6 else Chem.BondType(int(order))
        rdmol.AddBond(atom_idx[id(bond.atom1)], atom_idx[id(bond.atom2)], bond_type)
    Chem.SanitizeMol(rdmol)

    conf = Chem.Conformer(len(mol))
    for i, at in enumerate(mol.atoms):
        conf.SetAtomPosition(i, Geometry.Point3D(*at.coords))
    rdmol.AddConformer(conf, assignId=True)

    with open(filename, 'w') as f:
        f.write(Chem.MolToPDBBlock(rdmol))


#: A mapping of file extensions to a Callable for Molecule exporting
EXPORT_MAPPING: "MappingProxyType[str, MolExportFunc]" = MappingProxyType({
    'pdb': write_pdb,
    'xyz': Molecule.write,
    'mol': Molecule.write,
    'mol2': Molecule.write
})

#: The maximum length of a filename, in bytes, on most filesystems
MAX_FILENAME_BYTES = 255


def mol_to_file(
    mol_list: Iterable[Molecule],
    path: "None | str" = None,
    overwrite: bool = True,
    mol_format: Container[str] = ('xyz', 'pdb'),
) -> None:
    """Export all molecules in **mol_list** to .pdb, .xyz, .mol or .mol2 files.

    Parameters
    ----------
    mol_list: |list|_ [|plams.Molecule|_]
        An iterable consisting of PLAMS molecules.

    path : str
        Optional: The path to the directory where the molecules will be stored.
        Defaults to the current working directory if ``None``.

    overwrite : bool
        If previously generated files can be overwritten or not.
        If ``False``, files that already exist are left untouched.

    mol_format : |list|_ [|str|_]
        A list of strings with the to-be exported file types.
        Accepted values are ``"xyz"``, ``"pdb"``, ``"mol"`` and/or ``"mol2"``.

    Raises
    ------
    FileNotFoundError
        Raised **path** does not exist.

    NotADirectoryError
        Raised **path** is not a directory.

    Note
    ----
    Molecules whose name would make a filename longer than 255 bytes
    are written under a shortened name ending with a hash of the full name.

    """
    # Set the export path
    _path = path or os.getcwd()
    if not exists(_path):
        raise FileNotFoundError(f'{path} not found')
    elif not isdir(_path):
        raise NotADirectoryError(f'{path} is not a directory')

    export_dict = {k: v for k, v in EXPORT_MAPPING.items() if k in mol_format}
    if not export_dict:
        raise ValueError("No valid values found in the mol_format argument; accepted values are: "
                         "'xyz', 'pdb', 'mol' and/or 'mol2'")

    ext_len = max(len(ext) for ext in export_dict)
    for mol in mol_list:
        name = mol.properties.name
        short_name = shorten_filename(name, ext_len)
        if short_name != name:
            logger.warning(f"Filename too long: writing {name!r} as {short_name!r}")
        mol_path = join(_path, short_name)

        try:
            for ext, func in export_dict.items():
                filename = f'{mol_path}.{ext}'
                if not overwrite and isfile(filename):
                    continue
                func(mol, filename)
        except OSError as ex:
            # The filename itself was already shortened; the full path is too long
            if ex.errno == errno.ENAMETOOLONG:
                logger.error(f"Path too long: failed to create {filename!r}", exc_info=True)
            else:
                raise


def shorten_filename(name: str, ext_len: int = 4) -> str:
    """Shorten **name** if it would exceed :data:`MAX_FILENAME_BYTES` with an extension.

    Longer names are truncated and given a hash of the full name,
    so different long names still map to different files.

    Examples
    --------
    .. code:: python

        >>> from CAT.data_handling.mol_to_file import shorten_filename

        >>> shorten_filename('Cd68Se55')
        'Cd68Se55'

        >>> name = shorten_filename('C' * 300)
        >>> len(name)
        250
        >>> name[-17:]
        'CCCC_a5c66bb4b5e4'

    Parameters
    ----------
    name : str
        The filename, without its extension.
    ext_len : int
        The length of the longest extension that will be appended to **name**
        (excluding the ``"."``).

    Returns
    -------
    str
        **name** itself or, if too long, a shortened name ending with a 12-character hash.

    """
    max_len = MAX_FILENAME_BYTES - ext_len - 1
    name_bytes = name.encode()
    if len(name_bytes) <= max_len:
        return name

    digest = hashlib.sha256(name_bytes).hexdigest()[:12]
    prefix = name_bytes[:max_len - len(digest) - 1].decode(errors='ignore')
    return f'{prefix}_{digest}'
