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

__all__ = ["TBLiteJob", "TBLiteResults", "TEMPLATES", "check_tblite"]

#: The path to the script that runs the calculation of a :class:`TBLiteJob`
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")

#: The settings that the job presets of :mod:`CAT.jobs` add to the settings of a
#: :class:`TBLiteJob`, the counterpart of the QMFlows templates used for other job types
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


class TBLiteResults(Results):
    """The results of a :class:`TBLiteJob`."""

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
        return np.array(self._read()["charges"])

    def get_gradients(self) -> np.ndarray:
        """Return the gradient of the final geometry in Hartree/Bohr."""
        return np.array(self._read()["gradient"])


def _to_json(value: Any) -> Any:
    """Convert PLAMS Settings and NumPy objects into JSON-compatible ones."""
    if isinstance(value, Mapping):
        return {str(k).lower(): _to_json(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple, np.ndarray)):
        return [_to_json(i) for i in value]
    elif isinstance(value, np.generic):
        return value.item()
    return value


class TBLiteJob(SingleJob):
    """A PLAMS job running an extended tight-binding (xTB) calculation with tblite.

    The job runs GFN2-xTB by default, in a separate Python process
    (see :mod:`CAT.xtb.runner`).
    Its input settings are described in :ref:`xTB`.

    """

    _result_type = TBLiteResults

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
        ret = f'"{sys.executable}" "{RUNNER}" "{self._filename("inp")}"'
        if self.settings.runscript.stdout_redirect:
            ret += f' >"{self._filename("out")}"'
        return ret + "\n"

    def check(self) -> bool:
        """Return whether the calculation succeeded, including the geometry optimization."""
        try:
            return bool(self.results._read()["converged"])
        except Exception:
            return False
