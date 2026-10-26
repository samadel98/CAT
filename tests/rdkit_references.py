"""Reference geometries of the tests that depend on how RDKit lays out molecules.

:func:`nanoCAT.recipes.fast_bulk_workflow` takes its Cartesian coordinates from the 2D
layout that RDKit generates for a SMILES string, which changes between RDKit releases.
``test_bulk_recipe.py`` therefore takes its reference geometries from :data:`RDKIT_REF_PATH`:
the original ones (made with RDKit 2021.09) or those made with RDKit :data:`RDKIT_REF_VERSION`.

The layout of a few molecules also differs between platforms. ``mol_array.npy`` holds the
Linux geometries, and ``mol_array_<sys.platform>.json`` the coordinates of the molecules
whose layout differs on that platform, as printed by ``python tests/rdkit_references.py --diff``
(which CI runs when the tests fail).

After an RDKit update, regenerate the references by running this module from the repository
root on each platform (``python tests/rdkit_references.py tests/test_files/rdkit-<version>``),
check that the differences are only due to RDKit, and update :data:`RDKIT_REF_VERSION`.

"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import rdkit
import numpy as np
import pandas as pd
from packaging.version import Version
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import void as V
    from scm.plams import Molecule

__all__ = ["RDKIT_REF_PATH", "RDKIT_REF_VERSION", "RDKIT_OLD_REF", "MOL_ATOL", "load_mol_array"]

PATH = Path("tests") / "test_files"

#: The RDKit version (``year.month``) of the newest reference geometries.
RDKIT_REF_VERSION = "2026.09"

#: Whether the original reference geometries are used.
RDKIT_OLD_REF = Version(rdkit.__version__) < Version("2024.03")

RDKIT_REF_PATH = PATH if RDKIT_OLD_REF else PATH / f"rdkit-{RDKIT_REF_VERSION}"

#: The absolute tolerance of the interatomic distances (in Angstrom).
#: The coordinates pass through an RDKit MolBlock, which rounds them to 4 decimals.
MOL_ATOL = 1e-3


def load_mol_array(path: Path = RDKIT_REF_PATH, platform: str = sys.platform) -> NDArray[V]:
    """Load the reference geometries of ``test_bulk_recipe.py`` for **platform**."""
    ret = np.load(path / "mol_array.npy")
    json_file = path / f"mol_array_{platform}.json"
    if not json_file.is_file():
        return ret

    with open(json_file) as f:
        overrides: dict[str, dict[str, Any]] = json.load(f)
    for i, dct in overrides.items():
        n = len(dct["symbol"])
        ret["dist_mat"][int(i)] = np.nan
        ret["dist_mat"][int(i), :n, :n] = cdist(dct["coords"], dct["coords"])
        ret["symbol"][int(i)] = ""
        ret["symbol"][int(i), :n] = dct["symbol"]
    return ret


def _generate_mols() -> list[Molecule]:
    from nanoCAT.recipes import fast_bulk_workflow

    smiles = pd.read_csv(PATH / "bulk_input.csv")["smiles"]
    mol_list, _ = fast_bulk_workflow(smiles)
    return mol_list


def diff_mol_coords(ref: NDArray[V]) -> dict[str, dict[str, Any]]:
    """Return the coordinates of all molecules whose geometry differs from **ref**."""
    ret = {}
    for i, mol in enumerate(_generate_mols()):
        n = len(mol)
        symbol = [at.symbol for at in mol]
        dist_ref = ref["dist_mat"][i, :n, :n]
        if (
            symbol != ref["symbol"][i, :n].tolist()
            or not np.allclose(cdist(mol, mol), dist_ref, rtol=1e-7, atol=MOL_ATOL)
        ):
            coords = [list(at.coords) for at in mol]
            ret[str(i)] = {"smiles": mol.properties.smiles, "symbol": symbol, "coords": coords}
    return ret


def regenerate_mol_array(out: Path) -> None:
    """Regenerate ``mol_array.npy`` (on Linux) or ``mol_array_<sys.platform>.json``."""
    if sys.platform != "linux":
        overrides = diff_mol_coords(load_mol_array(out, platform="linux"))
        with open(out / f"mol_array_{sys.platform}.json", "w") as f:
            json.dump(overrides, f, indent=4)
            f.write("\n")
        return None

    mol_list = _generate_mols()
    n = max(len(mol) for mol in mol_list)
    ret = np.empty(len(mol_list), dtype=[("dist_mat", "f8", (n, n)), ("symbol", "U2", n)])
    ret["dist_mat"] = np.nan
    ret["symbol"] = ""
    for i, mol in enumerate(mol_list):
        ret["dist_mat"][i, :len(mol), :len(mol)] = cdist(mol, mol)
        ret["symbol"][i, :len(mol)] = [at.symbol for at in mol]
    np.save(out / "mol_array.npy", ret)


if __name__ == "__main__":
    if sys.argv[1:] == ["--diff"]:
        print(json.dumps(diff_mol_coords(load_mol_array(platform="linux")), indent=4))
    else:
        out = Path(sys.argv[1]) if len(sys.argv) > 1 else RDKIT_REF_PATH
        out.mkdir(parents=True, exist_ok=True)
        regenerate_mol_array(out)
