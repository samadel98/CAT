"""Tests for the xTB-based ligand descriptors: :mod:`nanoCAT.cdft` and :mod:`nanoCAT.xtb_solvation`."""  # noqa: E501

from __future__ import annotations

import os
import shutil
import importlib.util
from pathlib import Path
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
from scm.plams import Settings

from CAT.base import prep
from CAT.workflows import CDFT_XTB_COLUMNS, CDFT_XTB_IP, CDFT_XTB_EA, CDFT_XTB_ETA
from nanoCAT.cdft import get_xtb_descriptors

requires_tblite = pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="requires tblite"
)

PATH = Path("tests") / "test_files"
WORKDIR = PATH / "xtb_descriptors"


class TestGetXTBDescriptors:
    def test_values(self) -> None:
        ip, ea = 10.0, 1.0
        ret = get_xtb_descriptors(ip, ea)
        assert list(ret.index) == list(CDFT_XTB_COLUMNS)
        omega = 5.5**2 / 18
        ref = [
            ip, ea, -5.5, 5.5, 9.0, 1 / 9, omega,
            (ip - 3 * ea)**2 / (8 * (ip - ea)), (3 * ip - ea)**2 / (8 * (ip - ea)),
            (3 * ip + ea)**2 / 144, (ip + 3 * ea)**2 / 144,
            ((3 * ip + ea)**2 + (ip + 3 * ea)**2) / 144, -ip, -ea,
        ]
        np.testing.assert_allclose(ret.to_numpy(), ref)
        assert ret.iloc[7] == pytest.approx(omega - ea)
        assert ret.iloc[8] == pytest.approx(omega + ip)

    @pytest.mark.parametrize("ip,ea", [(1.0, 2.0), (np.nan, 1.0)], ids=["I<A", "nan"])
    def test_nan(self, ip: float, ea: float) -> None:
        ret = get_xtb_descriptors(ip, ea)
        assert np.isnan(ret[CDFT_XTB_ETA])
        np.testing.assert_array_equal(ret.iloc[[0, 1, -2, -1]], [ip, ea, -ip, -ea])


@requires_tblite
class TestWorkflow:
    SETTINGS = Settings({
        "path": str(WORKDIR),
        "input_cores": [str(PATH / "core" / "Cd68Se55.xyz")],
        "input_ligands": ["CC(=O)O", "CCCC(=O)O"],
        "optional": {
            "core": {"dummy": "Cl"},
            "qd": {"construct_qd": False},
            "ligand": {
                "split": True,
                "cdft": {"job1": "TBLiteJob"},
                "solvation": {"solvents": ["water", "toluene"]},
            },
        },
    })

    @pytest.fixture(scope="function", autouse=True)
    def setup_cat(self) -> Generator[None, None, None]:
        os.makedirs(WORKDIR, exist_ok=True)
        yield None
        shutil.rmtree(WORKDIR, ignore_errors=True)

    def test_pass(self) -> None:
        *_, ligand_df = prep(self.SETTINGS.copy())
        cdft = ligand_df[list(CDFT_XTB_COLUMNS)]
        assert not cdft.isna().any(axis=None)
        assert (cdft[CDFT_XTB_IP] > cdft[CDFT_XTB_EA]).all()

        solv = ligand_df["E_solv_xtb"]
        assert list(solv.columns) == ["water", "toluene"]
        assert not solv.isna().any(axis=None)
        assert (solv["water"] < 0).all()

        # Both are read back from the database
        *_, ligand_df2 = prep(self.SETTINGS.copy())
        pd.testing.assert_frame_equal(ligand_df2[list(CDFT_XTB_COLUMNS)], cdft)
        pd.testing.assert_frame_equal(ligand_df2["E_solv_xtb"], solv)
