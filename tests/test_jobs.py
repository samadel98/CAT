"""Tests for :mod:`CAT.jobs`."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from scm.plams import Cp2kResults, Units
from assertionlib import assertion

import CAT.jobs  # noqa: F401  # Adds `get_energy` to `Cp2kResults`

PATH = Path('tests') / 'test_files' / 'cp2k'


@pytest.mark.parametrize("filename,ref", [
    ("freq_2020.out", -36022.96379486),
    ("freq_2026.out", -45205.94680154),  # CP2K 2026 uses less whitespace after `VIB|`
], ids=["cp2k_2020", "cp2k_2026"])
def test_get_energy_freq(mocker, filename: str, ref: float) -> None:
    """Test :meth:`Cp2kResults.get_energy` for frequency analyses."""
    def grep_output(self, pattern: str = '', options: str = '') -> list[str]:
        with open(PATH / filename) as f:
            return [i.rstrip('\n') for i in f if pattern in i]

    mocker.patch.object(Cp2kResults, 'grep_output', grep_output)
    results = Cp2kResults.__new__(Cp2kResults)
    results.job = SimpleNamespace(name='freq', status='successful')

    energy = results.get_energy(job_preset='frequency analysis')
    assertion.isclose(energy, Units.convert(ref, 'kj/mol', 'Hartree'))

    energy_kcal = results.get_energy(unit='kcal/mol', job_preset='frequency analysis')
    assertion.isclose(energy_kcal, Units.convert(ref, 'kj/mol', 'kcal/mol'))
