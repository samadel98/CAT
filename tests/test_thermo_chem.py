"""Tests for :mod:`CAT.thermo_chem`."""

from os.path import join

import numpy as np

from scm.plams import Atom, Molecule, Units
from assertionlib import assertion

from CAT.thermo_chem import (get_entropy, get_thermo)

PATH = join('tests', 'test_files')
MOL = Molecule(join(PATH, 'Methanol.xyz'))  # Methanol; BP86/QZ4P
FREQ = np.load(join(PATH, 'freq.npy'))  # cm**-1


def test_get_entropy() -> None:
    """Tests for :func:`CAT.thermo_chem.get_entropy`."""
    ref1 = np.array([143.78052972, 79.61371901, 2308.88447934])
    s1 = get_entropy(MOL, FREQ)
    np.testing.assert_allclose(ref1, s1)

    # Test with a different temperature
    ref2 = np.array([149.8889032, 83.2787431, 2338.20466375])
    s2 = get_entropy(MOL, FREQ, T=400)
    np.testing.assert_allclose(ref2, s2)


def test_get_thermo() -> None:
    """Tests for :func:`CAT.thermo_chem.get_thermo`."""
    ref1 = {'E': 0.0,
            'U': 36.876336257248056,
            'H': 37.46882030779777,
            'S': 0.07589088733413608,
            'G': 14.841952249125093}
    thermo1 = get_thermo(MOL, FREQ)
    for k, v in thermo1.items():
        i, j = round(v, 8), round(ref1[k], 8)
        assertion.eq(i, j)

    # Test with E != 0.0
    ref2 = {'E': -692.08,
            'U': -655.203663742752,
            'H': -654.6111796922023,
            'S': 0.07589088733413608,
            'G': -677.238047750875}
    thermo2 = get_thermo(MOL, FREQ, -692.08)
    for k, v in thermo2.items():
        i, j = round(v, 8), round(ref2[k], 8)
        assertion.eq(i, j)

    # Test with a different unit (au)
    thermo3 = get_thermo(MOL, FREQ, unit='au')
    for k, v in thermo3.items():
        i, j = round(v, 8), round(Units.convert(ref1[k], 'kcal/mol', 'au'), 8)
        assertion.eq(i, j)

    # Test with a different temperature
    ref4 = {'E': 0.0,
            'U': 39.09512888121028,
            'H': 39.89000937834221,
            'S': 0.08286108516957914,
            'G': 6.745575310510546}
    thermo4 = get_thermo(MOL, FREQ, T=400)
    for k, v in thermo4.items():
        i, j = round(v, 8), round(ref4[k], 8)
        assertion.eq(i, j)

    # Test when exporting a single quantity
    ref5 = 14.841952249125093
    g = get_thermo(MOL, FREQ, export='G')
    i, j = round(g, 8), round(ref5, 8)
    assertion.eq(i, j)


def test_get_entropy_rotation() -> None:
    """Test the rotational entropy of a linear molecule and an atom, away from the origin."""
    hcl = Molecule()
    hcl.add_atom(Atom(symbol="H", coords=(10.0, 0.0, 0.0)))
    hcl.add_atom(Atom(symbol="Cl", coords=(11.2746, 0.0, 0.0)))
    S_rot = get_entropy(hcl, [2990.0])[1]
    assert abs(S_rot - 33.0) < 0.1  # J/mol/K

    thermo = get_thermo(hcl, [2990.0])
    assert np.isfinite(list(thermo.values())).all()

    atom = Molecule()
    atom.add_atom(Atom(symbol="Cl", coords=(1.0, 2.0, 3.0)))
    assert get_entropy(atom, [])[1] == 0.0
    assert np.isfinite(list(get_thermo(atom, []).values())).all()
