"""Reference geometries of the tests that depend on how RDKit embeds molecules.

Every RDKit release since 2024.03 generates different coordinates from SMILES strings,
so the tests in ``test_ligand_anchoring.py``, ``test_ligand_attach.py`` and
``test_entry_points.py`` take their reference geometries from :data:`RDKIT_REF_PATH`:
the original ones (made with RDKit 2021.03 - 2023.09) or those made with RDKit 2026.09.

After an RDKit update, regenerate the latter by running this module from the repository
root (``python tests/rdkit_references.py tests/test_files/rdkit-<version>``), check that
the differences are only due to RDKit, and update :data:`RDKIT_REF_VERSION`. The references
are made on Linux; ``XFAIL_PLATFORM`` in ``test_entry_points.py`` lists the ligands whose
geometry differs between platforms or even machines.

"""

import sys
import shutil
from pathlib import Path

import yaml
import h5py
import rdkit
import numpy as np
from packaging.version import Version
from scm.plams import Settings, Molecule

__all__ = ["RDKIT_REF_PATH", "RDKIT_REF_VERSION"]

PATH = Path("tests") / "test_files"

#: The RDKit version (``year.month``) of the newest reference geometries.
RDKIT_REF_VERSION = "2026.09"

if Version(rdkit.__version__) < Version("2024.03"):
    RDKIT_REF_PATH = PATH
else:
    RDKIT_REF_PATH = PATH / f"rdkit-{RDKIT_REF_VERSION}"

TMP_PATHS = [PATH / "ligand", PATH / "qd", PATH / "database"]

DIHEDRALS = {"dihed_45": 45, "dihed_45_deg": "45 deg", "dihed_180": 180.0}
ALLIGNMENTS = ("sphere", "surface", "sphere_invert", "surface_invert")
CORE_ANCHORS = {
    "HCl": ("Cd68Se55_HCl.pdb", {"group": "[H]Cl", "group_idx": 0, "remove": 0}),
    "formate": ("Cd68Cl26Se55__26_O=C[O-]@O2O3.pdb", {
        "group": "[H]C([O-])=O",
        "group_idx": [2, 3],
        "remove": [0, 1, 2, 3],
        "kind": "mean",
    }),
    "ethoxide": ("Cd68Cl26Se55__26_CC[O-]@O3.pdb", {
        "group": "[O-]C([H])([H])C([H])([H])[H]",
        "group_idx": 0,
        "remove": [0, 1, 2, 3, 4, 5, 6, 7],
        "kind": "first",
    }),
}
ANCHOR_OPTIONS = {
    "remove": dict(group="OC(=O)C", group_idx=1, remove=[0, 2, 3]),
    "kind_first": dict(group="OC(=O)C", group_idx=0, kind="FIRST"),
    "kind_mean": dict(group="OC(=O)C", group_idx=[0, 2], kind="MEAN"),
    "kind_mean_translate": dict(group="OC(=O)C", group_idx=[0, 2], kind="MEAN_TRANSLATE"),
    "angle": dict(group="OC(=O)C", group_idx=[0, 1, 2], angle_offset=45),
}


def _clean() -> None:
    for path in TMP_PATHS:
        shutil.rmtree(path, ignore_errors=True)


def _run_cat(yaml_name: str, **kwargs: object) -> Molecule:
    from CAT.base import prep
    from CAT.workflows import MOL

    with open(PATH / yaml_name, "r") as f:
        arg = Settings(yaml.load(f, Loader=yaml.FullLoader))
    arg.path = PATH
    for key, value in kwargs.items():
        arg.set_nested(tuple(key.split(".")), value)
    try:
        qd_df, _, _ = prep(arg)
        return qd_df[MOL].iloc[0]
    finally:
        _clean()


def _atoms_and_bonds(mol: Molecule) -> "tuple[np.ndarray, np.ndarray]":
    atoms = np.fromiter(
        ((at.symbol, at.coords) for at in mol), dtype=[("symbols", "S2"), ("coords", "f8", 3)],
    )
    try:
        mol.set_atoms_id()
        bonds = np.fromiter(
            ((b.atom1.id, b.atom2.id, b.order) for b in mol.bonds),
            dtype=[("atom1", "i8"), ("atom2", "i8"), ("order", "f8")],
        )
    finally:
        mol.unset_atoms_id()
    return atoms, bonds


def regenerate_ligand_attach(out: Path) -> None:
    """Regenerate the reference data of ``test_ligand_attach.py``."""
    for name, dihed in DIHEDRALS.items():
        qd = _run_cat("CAT_dihedral.yaml", **{"optional.ligand.anchor.dihedral": dihed})
        ar = np.fromiter(
            ((at.symbol, at.coords) for at in qd), dtype=[("symbols", "U2"), ("coords", "f8", 3)],
        )
        np.save(out / f"test_dihedral_{name}.npy", ar)

    with h5py.File(out / "test_allignment.hdf5", "w") as f:
        for allignment in ALLIGNMENTS:
            qd = _run_cat("CAT_allignment.yaml", **{"optional.core.allignment": allignment})
            atoms, bonds = _atoms_and_bonds(qd)
            f[f"TestAllignment/{allignment}/atoms"] = atoms
            f[f"TestAllignment/{allignment}/bonds"] = bonds

        for name, (core, anchor) in CORE_ANCHORS.items():
            kwargs = {"input_cores": [core], "optional.core.anchor": anchor}
            if name == "formate":
                kwargs["optional.core.allignment"] = "anchor"
            qd = _run_cat("CAT_allignment.yaml", **kwargs)
            atoms, bonds = _atoms_and_bonds(qd)
            f[f"TestCoreAnchor/{name}/atoms"] = atoms
            f[f"TestCoreAnchor/{name}/bonds"] = bonds


def regenerate_ligand_anchoring(out: Path) -> None:
    """Regenerate the ``TestFindSubstructure.test_options`` reference data."""
    from CAT.attachment.ligand_anchoring import find_substructure
    from CAT.attachment.ligand_opt import optimize_ligand
    from CAT.data_handling.anchor_parsing import parse_anchors
    from CAT.mol_utils import rdkit_props
    from CAT._compat import from_smiles

    with h5py.File(out / "test_ligand_anchoring.hdf5", "w") as f:
        for name, kwargs in ANCHOR_OPTIONS.items():
            mol = from_smiles("OC(=O)CCCCC")
            mol_out, = find_substructure(mol, parse_anchors(kwargs))
            optimize_ligand(mol_out)

            atoms = np.fromiter(
                ((rdkit_props(at).get("charge", 0.0), at.symbol) for at in mol_out),
                dtype=[("charge", "f8"), ("symbol", "S2")],
            )
            try:
                mol_out.set_atoms_id()
                bonds = np.fromiter(
                    ((b.atom1.id, b.atom2.id, b.order) for b in mol_out.bonds),
                    dtype=[("atom1", "i8"), ("atom2", "i8"), ("order", "i8")],
                )
            finally:
                mol_out.unset_atoms_id()

            group = f.create_group(f"TestFindSubstructure/test_options/{name}")
            group["atoms"] = atoms
            group["bonds"] = bonds
            group["coords"] = np.array(mol.as_array())


def regenerate_entry_points(out: Path) -> None:
    """Regenerate the ligand geometries of ``test_entry_points.py``."""
    from CAT.data_handling.entry_points import main

    (out / "ligand_ref").mkdir(exist_ok=True)
    try:
        main([str(PATH / "input2.yaml")])
        for file in (PATH / "ligand_ref").glob("*.xyz"):
            shutil.copy2(PATH / "ligand" / file.name, out / "ligand_ref" / file.name)
    finally:
        _clean()


def main(out: Path) -> None:
    """Regenerate all RDKit-dependent reference data in **out**."""
    out.mkdir(parents=True, exist_ok=True)
    regenerate_ligand_attach(out)
    regenerate_ligand_anchoring(out)
    regenerate_entry_points(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
