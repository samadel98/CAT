"""Compatibility utilities for supporting multiple NumPy and PLAMS versions."""

import os
import importlib
import itertools
from typing import Iterable, Optional, Type, Union

import numpy as np
import scm.plams.interfaces.molecule.rdkit as molkit
from rdkit import Chem
from rdkit.Chem import AllChem
from scm.plams import Atom, Molecule
from scm.plams.core.basejob import Job

__all__ = [
    "COPY_IF_NEEDED",
    "atoms_as_array",
    "RDKIT_PROPS_IN_SUBKEY",
    "get_conformation",
    "from_smiles",
    "readpdb",
    "ADFJob",
    "UFFJob",
    "BANDJob",
    "DFTBJob",
    "MOPACJob",
    "ReaxFFJob",
]

#: The value of the ``copy`` parameter in :func:`np.array` for only copying when needed.
#: This is ``False`` in NumPy 1.x, but NumPy 2 changed its meaning to "never copy"
#: in favor of ``None``.
COPY_IF_NEEDED: Optional[bool] = None if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else False


def atoms_as_array(atoms: Iterable[Atom]) -> np.ndarray:
    """Return the Cartesian coordinates of **atoms** as an :math:`n*3` array.

    Replaces ``Molecule.as_array(None, atom_subset=atoms)``,
    which stopped working once :meth:`Molecule.as_array` became a property in PLAMS 2024.
    """
    xyz = np.fromiter(itertools.chain.from_iterable(at.coords for at in atoms), dtype=float)
    return xyz.reshape(-1, 3)


def _rdkit_props_in_subkey() -> bool:
    """Check whether the PLAMS RDKit interface keeps formal charges in ``Atom.properties.rdkit``."""
    rdmol = Chem.MolFromSmiles("[Na+]")
    rdmol.AddConformer(Chem.Conformer(1))
    return "rdkit" in molkit.from_rdmol(rdmol)[1].properties


#: Whether PLAMS keeps the RDKit-related properties of atoms and bonds (``charge``, ``pdb_info``
#: and ``stereo``) in a ``rdkit`` sub-:class:`~scm.plams.Settings`, as PLAMS >= 2024 does.
RDKIT_PROPS_IN_SUBKEY: bool = _rdkit_props_in_subkey()


def get_conformation(rdmol: Chem.Mol) -> Molecule:
    """Embed a single conformer of **rdmol** and convert it into a PLAMS molecule.

    Mirrors ``molkit.get_conformations(rdmol, 1, rms=0.1)`` of PLAMS < 2024. Later PLAMS versions
    embed molecules with different settings, which would make CAT's geometries depend on the
    PLAMS version.
    """
    kwargs = dict(pruneRmsThresh=0.1, randomSeed=1, enforceChirality=False)
    try:
        AllChem.EmbedMultipleConfs(rdmol, 1, **kwargs)
    except Exception:
        # `useRandomCoords=True` prevents a (poorly documented) crash for large systems
        AllChem.EmbedMultipleConfs(rdmol, 1, useRandomCoords=True, **kwargs)
    return molkit.from_rdmol(rdmol)


def from_smiles(smiles: str) -> Molecule:
    """Create a PLAMS molecule from **smiles** like ``scm.plams.from_smiles()`` of PLAMS < 2024.

    See :func:`get_conformation`.
    """
    smiles = Chem.CanonSmiles(smiles.split()[0])
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdmol.SetProp("smiles", smiles)
    return get_conformation(rdmol)


def readpdb(filename: Union[str, "os.PathLike[str]"]) -> Molecule:
    """Read a PDB file into a PLAMS molecule, reading bonds only from its ``CONECT`` records.

    Unlike ``molkit.readpdb()`` of PLAMS >= 2024, which lets RDKit guess (and maybe reject) bonds.
    """
    rdmol = Chem.MolFromPDBFile(os.fspath(filename), removeHs=False, proximityBonding=False)
    return molkit.from_rdmol(rdmol)


def _import_legacy_job(module: str, name: str) -> Optional[Type[Job]]:
    """Import a job class that newer PLAMS versions removed, returning ``None`` if it is absent."""
    try:
        return getattr(importlib.import_module(f"scm.plams.interfaces.adfsuite.{module}"), name)
    except (ImportError, AttributeError):
        return None


# PLAMS >= 2024 removed the job classes of the pre-AMS engines, which now all run via `AMSJob`
ADFJob = _import_legacy_job("adf", "ADFJob")
UFFJob = _import_legacy_job("uff", "UFFJob")
BANDJob = _import_legacy_job("band", "BANDJob")
DFTBJob = _import_legacy_job("dftb", "DFTBJob")
MOPACJob = _import_legacy_job("mopac", "MOPACJob")
ReaxFFJob = _import_legacy_job("reaxff", "ReaxFFJob")
