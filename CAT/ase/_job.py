"""The PLAMS job class for running calculations with an ASE calculator."""

from __future__ import annotations

import os
import json
import importlib.util

from ..logger import logger
from ..mol_utils import rdkit_props
from ..xtb._job import RunnerJob, RunnerResults, _to_json

__all__ = ["ASEJob", "ASEResults", "check_ase"]

#: The path to the script that runs the calculation of an :class:`ASEJob`
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")


def check_ase() -> None:
    """Raise a :exc:`ModuleNotFoundError` if ASE is not installed."""
    if importlib.util.find_spec("ase") is None:
        msg = "ASEJob requires the ase package; install it with `pip install ase`"
        logger.error(msg)
        raise ModuleNotFoundError(msg, name="ase")


def _key(key: object) -> str:
    return str(key).lower().replace("-", "_")


class ASEResults(RunnerResults):
    """The results of an :class:`ASEJob`."""


class ASEJob(RunnerJob):
    """A PLAMS job running a calculation with an `ASE <https://ase-lib.org>`_ calculator.

    Any ASE calculator can be used, *e.g.* machine-learned interatomic potentials such as
    MACE, in a separate Python process (see :mod:`CAT.ase.runner`).
    The geometry optimization and frequency analysis are those of :class:`~CAT.xtb.TBLiteJob`.
    Its input settings are described in :ref:`ASE`.

    """

    _result_type = ASEResults
    RUNNER = RUNNER

    def get_input(self) -> str:
        """Return the JSON input of the runner script.

        The input consists of the ``input`` settings and a list of atoms, each atom
        consisting of its symbol, atomic number, mass and Cartesian coordinates (Angstrom).
        The keys of ``input.calculator_kwargs`` keep their case.

        """
        settings = self.settings.input
        inp = _to_json({k: v for k, v in settings.items() if _key(k) != "calculator_kwargs"})
        for k, v in settings.items():
            if _key(k) == "calculator_kwargs" and v is not None:
                inp["calculator_kwargs"] = _to_json(v, lower=False)

        formal_charges = [rdkit_props(at).get("charge", 0) for at in self.molecule]
        inp.setdefault("charge", int(sum(formal_charges)))
        inp["formal_charges"] = formal_charges
        atoms = ",\n".join(
            f"        {json.dumps([at.symbol, at.atnum, at.mass, *at.coords])}"
            for at in self.molecule
        )
        return f'{{\n    "input": {json.dumps(inp)},\n    "atoms": [\n{atoms}\n    ]\n}}\n'
