"""Tests for :mod:`nanoCAT.recipes.multi_lig_job`."""

from pathlib import Path

import numpy as np

from nanoCAT.recipes.multi_lig_job import _concatenate_prm

_NONBONDED = """NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -
cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5
"""

PRM1 = f"""ATOMS
MASS -1 Cd 112.411
MASS -1 Se 78.971

{_NONBONDED}Cd 0.0 -0.1 1.2
Se 0.0 -0.2 2.2

END
"""

PRM2 = f"""ATOMS
MASS -1 Cd 112.411
MASS -1 O 15.999

BONDS
O Cd 100.0 2.2

{_NONBONDED}Cd 0.0 -0.1 1.2
O 0.0 -0.3 1.7

END
"""


def test_concatenate_prm(tmp_path: Path) -> None:
    files = [tmp_path / "1.prm", tmp_path / "2.prm"]
    for file, content in zip(files, [PRM1, PRM2]):
        file.write_text(content)

    prm = _concatenate_prm(files)
    assert prm.atoms is not None and prm.nonbonded is not None and prm.bonds is not None
    assert prm.atoms.index.tolist() == ["Cd", "Se", "O"]
    assert prm.nonbonded.index.tolist() == ["Cd", "Se", "O"]
    np.testing.assert_allclose(prm.nonbonded[2], [-0.1, -0.2, -0.3])
    assert len(prm.bonds) == 1
