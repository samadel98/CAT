"""Tests for :mod:`CAT.report`."""

from __future__ import annotations

import json
import html
from pathlib import Path

import pandas as pd
import pytest
from schema import SchemaError
from scm.plams import Settings

from CAT._compat import from_smiles
from CAT.base import prep
from CAT.report import write_report
from CAT.workflows import MOL
from CAT.data_handling.validation_schemas import report_schema

PATH = Path("tests") / "test_files"


def _read_data(filename: Path) -> dict:
    """Return the JSON data embedded in the report."""
    text = filename.read_text(encoding="utf-8")
    start = text.index('<script type="application/json" id="cat-data">') + 46
    end = text.index("</script>", start)
    return json.loads(text[start:end])


@pytest.mark.parametrize("value,ref", [
    (True, "report.html"),
    (False, False),
    (None, False),
    ("run.html", "run.html"),
    (Path("sub") / "run.html", str(Path("sub") / "run.html")),
])
def test_schema(value: object, ref: object) -> None:
    assert report_schema.validate(value) == ref


@pytest.mark.parametrize("value", [1, "", " ", ["report.html"]])
def test_schema_raise(value: object) -> None:
    with pytest.raises(SchemaError):
        report_schema.validate(value)


def test_write_report(tmp_path: Path) -> None:
    mol = from_smiles("CC(=O)[O-]")
    columns = pd.MultiIndex.from_tuples(
        [("mol", ""), ("formula", ""), ("cone_angle", "dist=0.0"), ("job_settings_cdft", ""),
         ("note", "")],
        names=["index", "sub index"],
    )
    index = pd.MultiIndex.from_tuples([("CC(=O)[O-]", "O4"), ("<b>", "C1")],
                                      names=["smiles", "anchor"])
    df = pd.DataFrame(
        [[mol, "C2H3O2", 82.9374, ["s1"], "</script><script>alert(1)</script>"],
         [None, "C1", float("nan"), ["s2"], True]],
        index=index, columns=columns,
    )
    filename = tmp_path / "report.html"
    write_report(filename, ligand_df=df, title="Test & report", elapsed=1.25)
    text = filename.read_text(encoding="utf-8")

    assert "<title>Test &amp; report</title>" in text
    assert "1.2 s" in text
    assert "<th>cone_angle: dist=0.0</th>" in text and "job_settings" not in text
    assert "<td>82.9374</td>" in text and "<td>yes</td>" in text
    assert html.escape("</script><script>alert(1)</script>") in text
    assert "<td>&lt;b&gt;</td>" in text
    assert text.count("<b>2</b><span>ligands</span>") == 1
    assert "quantum dots" not in text  # No quantum dot dataframe

    # Only the row with a molecule gets a 3D view
    assert text.count('class="view"') == 1
    structures = _read_data(filename)["structures"]
    assert list(structures) == ["ligand-0"]
    ligand = structures["ligand-0"]
    assert ligand["name"] == "ligand: CC(=O)[O-] | O4"
    assert ligand["symbols"] == [at.symbol for at in mol]
    assert len(ligand["xyz"]) == 3 * len(mol)
    assert len(ligand["bonds"]) == len(mol.bonds)
    assert all(0 <= i < len(mol) for bond in ligand["bonds"] for i in bond)


def test_write_report_empty(tmp_path: Path) -> None:
    filename = tmp_path / "report.html"
    write_report(filename)
    text = filename.read_text(encoding="utf-8")
    assert "<p>No molecules.</p>" in text
    assert _read_data(filename)["structures"] == {}


def test_prep(tmp_path: Path) -> None:
    s = Settings({
        "path": str(tmp_path),
        "input_cores": [str(PATH / "core" / "Cd68Se55.xyz")],
        "input_ligands": ["CC(=O)O", "CCCC(=O)O"],
        "optional": {
            "report": "run.html",
            "core": {"dummy": "Cl"},
            "ligand": {"split": True},
            "database": {"read": False, "write": False},
        },
    })
    qd_df, core_df, ligand_df = prep(s)
    structures = _read_data(tmp_path / "run.html")["structures"]
    assert len(qd_df) == len(ligand_df) == 2 and len(core_df) == 1
    assert sorted(structures) == ["core-0", "ligand-0", "ligand-1", "qd-0", "qd-1"]
    assert len(structures["qd-1"]["symbols"]) == len(qd_df[MOL].iloc[1])
