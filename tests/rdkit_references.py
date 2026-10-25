"""Reference geometries of the tests that depend on how RDKit lays out molecules.

:func:`nanoCAT.recipes.fast_bulk_workflow` takes its Cartesian coordinates from the 2D
layout that RDKit generates for a SMILES string, which changes between RDKit releases.
``test_bulk_recipe.py`` therefore takes its reference geometries from :data:`RDKIT_REF_PATH`:
the original ones (made with RDKit 2021.09) or those made with RDKit :data:`RDKIT_REF_VERSION`.

After an RDKit update, regenerate the latter by running this module from the repository
root (``python tests/rdkit_references.py tests/test_files/rdkit-<version>``), check that
the differences are only due to RDKit, and update :data:`RDKIT_REF_VERSION`.

"""

from __future__ import annotations

import sys
from pathlib import Path

import rdkit
import numpy as np
import pandas as pd
from packaging.version import Version
from scipy.spatial.distance import cdist

__all__ = ["RDKIT_REF_PATH", "RDKIT_REF_VERSION", "RDKIT_OLD_REF"]

PATH = Path("tests") / "test_files"

#: The RDKit version (``year.month``) of the newest reference geometries.
RDKIT_REF_VERSION = "2026.09"

#: Whether the original reference geometries are used.
RDKIT_OLD_REF = Version(rdkit.__version__) < Version("2024.03")

if RDKIT_OLD_REF:
    RDKIT_REF_PATH = PATH
else:
    RDKIT_REF_PATH = PATH / f"rdkit-{RDKIT_REF_VERSION}"


def regenerate_mol_array(out: Path) -> None:
    """Regenerate ``mol_array.npy``, the reference geometries of ``test_bulk_recipe.py``."""
    from nanoCAT.recipes import fast_bulk_workflow

    smiles = pd.read_csv(PATH / "bulk_input.csv")["smiles"]
    mol_list, _ = fast_bulk_workflow(smiles)

    n = max(len(mol) for mol in mol_list)
    ret = np.empty(len(mol_list), dtype=[("dist_mat", "f8", (n, n)), ("symbol", "U2", n)])
    ret["dist_mat"] = np.nan
    ret["symbol"] = ""
    for i, mol in enumerate(mol_list):
        ret["dist_mat"][i, :len(mol), :len(mol)] = cdist(mol, mol)
        ret["symbol"][i, :len(mol)] = [at.symbol for at in mol]
    np.save(out / "mol_array.npy", ret)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RDKIT_REF_PATH
    out.mkdir(parents=True, exist_ok=True)
    regenerate_mol_array(out)
