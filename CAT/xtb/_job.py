"""The PLAMS job and results classes for running xTB calculations with tblite."""

from __future__ import annotations

import os
import sys
import json
import importlib.util
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

import numpy as np
from scm.plams import Molecule, Results, ResultsError, Settings, Units
from scm.plams.core.basejob import SingleJob

from ..logger import logger
from ..mol_utils import rdkit_props

__all__ = [
    "RunnerJob", "RunnerResults", "TBLiteJob", "TBLiteResults", "TEMPLATES", "check_tblite"
]

#: The path to the script that runs the calculation of a :class:`TBLiteJob`
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")

#: The settings that the job presets of :mod:`CAT.jobs` add to the settings of a
#: :class:`TBLiteJob` (or any other :class:`RunnerJob`),
#: the counterpart of the QMFlows templates used for other job types
TEMPLATES: Mapping[str, Settings] = MappingProxyType({
    "singlepoint": Settings({"task": "singlepoint"}),
    "geometry": Settings({"task": "geometry"}),
    "freq": Settings({"task": "frequencies"}),
})


def check_tblite() -> None:
    """Raise a :exc:`ModuleNotFoundError` if tblite is not installed."""
    if importlib.util.find_spec("tblite") is None:
        msg = ("TBLiteJob requires the tblite package; install it with "
               "`pip install tblite` or `pip install 'nlesc-CAT[xtb]'`")
        logger.error(msg)
        raise ModuleNotFoundError(msg, name="tblite")


class RunnerResults(Results):
    """The results of a :class:`RunnerJob`, read from the JSON output of its runner script."""

    def _read(self) -> dict[str, Any]:
        with open(self["$JN.json"]) as f:
            ret: dict[str, Any] = json.load(f)
        return ret

    def get_energy(self, unit: str = "Hartree") -> float:
        """Return the energy, by default in Hartree."""
        return float(Units.convert(self._read()["energy"], "Hartree", unit))

    def get_main_molecule(self) -> Molecule:
        """Return a copy of the input molecule with the final coordinates."""
        mol = self.job.molecule.copy()
        mol.from_array(self._read()["positions"])
        return mol

    def get_frequencies(self, unit: str = "cm-1") -> np.ndarray:
        """Return the vibrational frequencies, by default in cm**-1.

        Imaginary frequencies are returned as negative numbers.

        """
        freqs = self._read()["frequencies"]
        if freqs is None:
            raise ResultsError(f"{self.job.name}: frequencies are only available for the "
                               "'frequencies' task")
        return np.asarray(Units.convert(np.array(freqs), "cm-1", unit))

    def get_charges(self) -> np.ndarray:
        """Return the atomic partial charges."""
        charges = self._read()["charges"]
        if charges is None:
            raise ResultsError(f"{self.job.name}: the calculator does not compute charges")
        return np.array(charges, dtype=np.float64)

    def get_gradients(self) -> np.ndarray:
        """Return the gradient of the final geometry in Hartree/Bohr."""
        return np.array(self._read()["gradient"])


class TBLiteResults(RunnerResults):
    """The results of a :class:`TBLiteJob`."""

    def get_solvation_energies(self, unit: str = "Hartree") -> dict[str, float]:
        """Return the solvation free energy in each solvent, by default in Hartree.

        Only available for the ``solvation`` task, which computes the energy in each of the
        ``input.solvents`` relative to the gas-phase energy, at the same geometry.

        """
        solv: None | dict[str, float] = self._read().get("solvation_energies")
        if solv is None:
            raise ResultsError(f"{self.job.name}: solvation energies are only available for the "
                               "'solvation' task")
        return {k: float(Units.convert(v, "Hartree", unit)) for k, v in solv.items()}

    def _get_ipea(self, key: str, unit: str) -> float:
        dct = self._read()
        if dct["task"] != "ipea":
            raise ResultsError(f"{self.job.name}: ionization potentials and electron affinities "
                               "are only available for the 'ipea' task")
        value: None | float = dct[key]
        return np.nan if value is None else float(Units.convert(value, "Hartree", unit))

    def get_ionization_potential(self, unit: str = "Hartree") -> float:
        """Return the vertical ionization potential, by default in Hartree.

        Only available for the ``ipea`` task; ``nan`` if the SCC of the cation did not converge.

        """
        return self._get_ipea("ionization_potential", unit)

    def get_electron_affinity(self, unit: str = "Hartree") -> float:
        """Return the vertical electron affinity, by default in Hartree.

        Only available for the ``ipea`` task; ``nan`` if the SCC of the anion did not converge.

        """
        return self._get_ipea("electron_affinity", unit)


def _to_json(value: Any, lower: bool = True) -> Any:
    """Convert PLAMS Settings and NumPy objects into JSON-compatible ones.

    The keys of mappings are converted into lowercase strings if **lower** is true.

    """
    if isinstance(value, Mapping):
        return {(str(k).lower() if lower else str(k)): _to_json(v, lower)
                for k, v in value.items()}
    elif isinstance(value, (list, tuple, np.ndarray)):
        return [_to_json(i, lower) for i in value]
    elif isinstance(value, np.generic):
        return value.item()
    return value


class RunnerJob(SingleJob):
    """A PLAMS job running a calculation with a Python script, :attr:`RUNNER`.

    The script is run with the Python interpreter of CAT. It reads the JSON input written
    by :meth:`get_input`, and writes the results to ``JOBNAME.json`` and the final geometry
    to ``JOBNAME.xyz``.

    """

    _result_type: type[RunnerResults] = RunnerResults

    #: The path to the script that runs the calculation
    RUNNER: str

    def get_input(self) -> str:
        """Return the JSON input of the runner script.

        The input consists of the ``input`` settings and a list of atoms, each atom
        consisting of its symbol, atomic number, mass and Cartesian coordinates (Angstrom).

        """
        inp = _to_json(self.settings.input)
        if "charge" not in inp:
            inp["charge"] = int(sum(rdkit_props(at).get("charge", 0) for at in self.molecule))
        atoms = ",\n".join(
            f"        {json.dumps([at.symbol, at.atnum, at.mass, *at.coords])}"
            for at in self.molecule
        )
        return f'{{\n    "input": {json.dumps(inp)},\n    "atoms": [\n{atoms}\n    ]\n}}\n'

    def get_runscript(self) -> str:
        """Return the command that runs the runner script with this interpreter."""
        ret = f'"{sys.executable}" "{self.RUNNER}" "{self._filename("inp")}"'
        if self.settings.runscript.stdout_redirect:
            ret += f' >"{self._filename("out")}"'
        return ret + "\n"

    def check(self) -> bool:
        """Return whether the calculation succeeded, including the geometry optimization."""
        try:
            return bool(self.results._read()["converged"])
        except Exception:
            return False


class TBLiteJob(RunnerJob):
    """A PLAMS job running an extended tight-binding (xTB) calculation with tblite.

    The job runs GFN2-xTB by default, in a separate Python process
    (see :mod:`CAT.xtb.runner`).
    Its input settings are described in :ref:`xTB`.

    """

    _result_type = TBLiteResults
    RUNNER = RUNNER
