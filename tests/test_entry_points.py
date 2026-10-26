"""Tests for :mod:`CAT.data_handling.entry_points`."""

import os
import sys
import shutil
import warnings
from pathlib import Path
from collections.abc import Generator

import rdkit
import pytest
from scm.plams import Molecule
from packaging.version import Version

from CAT.data_handling.entry_points import main
from CAT.test_utils import assert_mol_allclose

from rdkit_references import RDKIT_REF_PATH


PATH = Path('tests') / 'test_files'

LIG_PATH = PATH / 'ligand'
QD_PATH = PATH / 'qd'
DB_PATH = PATH / 'database'
PATH_REF = RDKIT_REF_PATH / 'ligand_ref'

#: Ligands whose geometry depends on the platform (or even the machine), per :data:`sys.platform`
XFAIL_PLATFORM = {
    "linux": {
        "CCCCCCCCP[CCCCCCCC]CCCCCCCC@P25.xyz",
        "CCCCCCCCP[=O][CCCCCCCC]CCCCCCCC@O26.xyz",
        "O=C[[O-]]c1ccc[S[=O][=O]CCCC[F][F]F]o1@O15.xyz",
        "O=C[COc1ccc[Cl]cc1Cl]Nc1cc[C[=O][O-]]ccc1F@O21.xyz",
    },
    "win32": {
        "CCCCCCCCP[CCCCCCCC]CCCCCCCC@P25.xyz",
        "CCCCCCCCP[=O][CCCCCCCC]CCCCCCCC@O26.xyz",
        "O=C[COc1ccc[Cl]cc1Cl]Nc1cc[C[=O][O-]]ccc1F@O21.xyz",
    },
    "darwin": {
        "O=C[[O-]]c1ccc[S[=O][=O]CCCC[F][F]F]o1@O15.xyz",
        "O=C[COc1ccc[Cl]cc1Cl]Nc1cc[C[=O][O-]]ccc1F@O21.xyz",
    },
}
if RDKIT_REF_PATH != PATH:
    # The newer references were made on Linux
    XFAIL_PLATFORM["win32"] |= {
        "CCCCCCCCC[CCCCCC]C[=O]O@O17.xyz",
        # Same geometry, but the atoms of the benzene ring are in a different order
        "O=C[O]c1ccccc1@O8.xyz",
    }
    XFAIL_PLATFORM["darwin"] |= {
        "CCCCCCCCC[CCCCCC]C[=O]O@O17.xyz",
        "cis-CCCCCCCCC=CCCCCCCCC[=O]O@O19.xyz",
        "CCCCCCCCP[=O][CCCCCCCC]CCCCCCCC@O26.xyz",
        "CCCCCCCCPCCCCCCCC@P17.xyz",
        "CCCCCCCCP[CCCCCCCC]CCCCCCCC@P25.xyz",
    }


class TestMain:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def run_cat(cls) -> Generator[None, None, None]:
        # Setup
        filename = str(PATH / 'input2.yaml')
        main([filename])

        yield None

        # Teardown
        files = [LIG_PATH, QD_PATH, DB_PATH]
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f)

    PATH_DICT = {
        os.path.splitext(f)[0]: f for f in os.listdir(PATH_REF) if f.endswith('xyz')
    }

    def test_raise(self) -> None:
        filename = str(PATH / 'input2.yaml')
        with pytest.raises(FileNotFoundError), warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="divide by zero encountered in double_scalars",
                category=RuntimeWarning,
            )
            main([f'{filename}bob'])

    @pytest.mark.parametrize("filename", PATH_DICT.values(), ids=PATH_DICT.keys())
    @pytest.mark.xfail(
        Version(rdkit.__version__) < Version("2021.03.4"),
        reason="requires rdkit >= 2021.03.4",
    )
    def test_mol(self, filename: str) -> None:
        mol = Molecule(LIG_PATH / filename)
        mol_ref = Molecule(PATH_REF / filename)

        platform = "linux" if sys.platform.startswith("linux") else sys.platform
        if filename in XFAIL_PLATFORM.get(platform, ()):
            pytest.xfail("platform dependant geometry")

        # Coordinates
        assert_mol_allclose(mol, mol_ref, rtol=0, atol=10**-2)
