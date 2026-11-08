"""Run the calculation of an :class:`~CAT.ase.ASEJob`.

The runscript of an :class:`~CAT.ase.ASEJob` runs this file as a script::

    python runner.py JOBNAME.in

It reads the JSON input written by :meth:`ASEJob.get_input()<CAT.ase.ASEJob.get_input>`,
writes the results to ``JOBNAME.json`` and the final geometry to ``JOBNAME.xyz``.
It imports NumPy, ASE and the package of the calculator, but neither CAT nor PLAMS:
the geometry optimization and the frequency analysis are those of :mod:`CAT.xtb.runner`,
which is loaded from its file.

Units
-----
The input and output coordinates are in Angstrom, the energy in Hartree,
the gradient in Hartree/Bohr and the frequencies in cm**-1, like those of
:mod:`CAT.xtb.runner`; the ASE calculator itself works in eV and Angstrom.

"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

# Python puts the directory of this script first in `sys.path`:
# remove it so that nothing in it can shadow an installed package
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [i for i in sys.path if os.path.abspath(i or os.curdir) != _HERE]

import time
import importlib
import importlib.util
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

import numpy as np


def _load_xtb_runner() -> ModuleType:
    """Load :mod:`CAT.xtb.runner` from its file, without importing CAT."""
    name = "CAT.xtb.runner"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(_HERE), "xtb", "runner.py")
    spec = importlib.util.spec_from_file_location("_cat_xtb_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xtb = _load_xtb_runner()

__all__ = ["CALCULATORS", "DEFAULTS", "main", "run"]

#: Aliases of ASE calculators: the import path of a calculator class or of a function
#: that returns a calculator
CALCULATORS: Mapping[str, str] = {
    "emt": "ase.calculators.emt.EMT",
    "lj": "ase.calculators.lj.LennardJones",
    "orca": "ase.calculators.orca.ORCA",
    "tblite": "tblite.ase.TBLite",
    "mace_mp": "mace.calculators.mace_mp",
    "mace_off": "mace.calculators.mace_off",
}

#: The default input settings.
#: Their keys, including those of the nested dictionaries, are the only valid input keys;
#: ``calculator_kwargs`` is passed as is to the calculator.
DEFAULTS: Mapping[str, Any] = {
    "task": "singlepoint",
    "calculator": None,
    "calculator_kwargs": None,
    "charge": 0,
    "uhf": None,
    "formal_charges": None,
    "geometry": {
        "max_iter": 1000,
        "gradient_tol": 1e-3,
        "max_step": 0.2,
        "memory": 100,
    },
    "frequencies": {
        "step": 0.0025,
    },
    "constraints": {
        "atom": [],
    },
}

TASKS = frozenset({"singlepoint", "geometry", "frequencies"})

Log = Callable[[str], None]
_print: Log = xtb._print


def get_calculator(name: str, kwargs: Mapping[str, Any] | None = None) -> Any:
    """Return the ASE calculator **name**, constructed with the keyword arguments **kwargs**.

    **name** is one of the :data:`CALCULATORS` aliases or the import path of a calculator
    class or of a function that returns a calculator, *e.g.*
    ``"ase.calculators.lj.LennardJones"`` or ``"mace.calculators:mace_mp"``.

    """
    if not isinstance(name, str):
        raise TypeError("The 'calculator' setting is required: the name or import path "
                        "of an ASE calculator")
    path = CALCULATORS.get(name.lower(), name)
    module_name, sep, attr = path.rpartition(":") if ":" in path else path.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid calculator: {name!r}; expected one of {sorted(CALCULATORS)} "
                         "or an import path such as 'ase.calculators.emt.EMT'")
    func = getattr(importlib.import_module(module_name), attr)
    return func(**(kwargs or {}))


class ASEEngine:
    """Compute the energy and gradient of a molecule with an ASE calculator, in atomic units.

    It has the interface of :class:`CAT.xtb.runner.Engine`, so that it can be used by
    the geometry optimization and frequency analysis of :mod:`CAT.xtb.runner`.

    The total charge and spin multiplicity are stored in ``atoms.info["charge"]`` and
    ``atoms.info["spin"]`` (as used by *e.g.* FAIR-chem), and in the initial charges and
    magnetic moments of the atoms (as used by *e.g.* tblite).

    """

    def __init__(self, numbers: np.ndarray, positions: np.ndarray,
                 inp: Mapping[str, Any]) -> None:
        from ase import Atoms
        from ase.units import Hartree

        #: Hartree in eV, as used by ASE
        self.hartree: float = Hartree
        self.charge = inp["charge"]
        self.uhf = inp["uhf"]
        if self.uhf is None:  # The lowest spin state for this number of electrons
            self.uhf = int(round(numbers.sum() - self.charge)) % 2

        n = len(numbers)
        formal_charges = inp["formal_charges"]
        if formal_charges is None or not np.isclose(sum(formal_charges), self.charge):
            formal_charges = np.full(n, self.charge / max(n, 1))

        self.atoms = Atoms(numbers=numbers, positions=positions * xtb.BOHR)
        self.atoms.info["charge"] = self.charge
        self.atoms.info["spin"] = self.uhf + 1
        self.atoms.set_initial_charges(formal_charges)
        if self.uhf:
            self.atoms.set_initial_magnetic_moments(np.full(n, self.uhf / n))
        self.atoms.calc = get_calculator(inp["calculator"], inp["calculator_kwargs"])
        self.count = 0

    def __call__(self, positions: np.ndarray, keep: bool = True) -> tuple[float, np.ndarray]:
        """Return the energy (Hartree) and gradient (Hartree/Bohr) at **positions** (Bohr)."""
        self.atoms.positions = positions * xtb.BOHR
        energy = self.atoms.get_potential_energy() / self.hartree
        gradient = -self.atoms.get_forces() / self.hartree * xtb.BOHR
        self.count += 1
        return float(energy), np.array(gradient, dtype=np.float64)

    def charges(self) -> list[float] | None:
        """Return the atomic partial charges, or ``None`` if the calculator does not compute them."""  # noqa: E501
        try:
            return np.asarray(self.atoms.get_charges(), dtype=np.float64).tolist()
        except Exception:
            return None


def run(data: Mapping[str, Any], log: Log = _print) -> dict[str, Any]:
    """Run the calculation described by the JSON input **data** and return its results.

    **data** consists of the ``"input"`` settings (see :data:`DEFAULTS`) and the ``"atoms"``,
    a list with the symbol, atomic number, mass and Cartesian coordinates (Angstrom)
    of each atom.

    """
    inp = xtb.parse_input(data.get("input", {}), DEFAULTS, tasks=TASKS)
    task = inp["task"].lower()
    symbols = [at[0] for at in data["atoms"]]
    numbers = np.array([at[1] for at in data["atoms"]], dtype=np.int64)
    masses = np.array([at[2] for at in data["atoms"]], dtype=np.float64)
    positions = np.array([at[3:6] for at in data["atoms"]], dtype=np.float64) / xtb.BOHR
    free = xtb._get_free(inp["constraints"]["atom"], len(numbers))

    start = time.perf_counter()
    engine = ASEEngine(numbers, positions, inp)
    log(f"ASE {inp['calculator']} {task}: {len(numbers)} atoms ({np.count_nonzero(~free)} "
        f"frozen), charge {engine.charge}, {engine.uhf} unpaired electron(s)")

    converged = True
    steps = 0
    if task == "geometry":
        positions, energy, gradient, converged, steps = xtb.optimize(
            engine, positions, free, log=log, **inp["geometry"]
        )
    else:
        energy, gradient = engine(positions)
    charges = engine.charges()

    freqs = None
    if task == "frequencies":
        hess = xtb.hessian(engine, positions, free, step=inp["frequencies"]["step"])
        freqs = xtb.frequencies(hess, masses[free], positions if free.all() else None)
        n_imag = np.count_nonzero(freqs < 0)
        log(f"{len(freqs)} vibrational frequencies, {n_imag} imaginary")

    log(f"Final energy: {energy:.10f} Hartree "
        f"({engine.count} calculations in {time.perf_counter() - start:.2f} s)")
    return {
        "task": task,
        "method": inp["calculator"],
        "charge": engine.charge,
        "uhf": engine.uhf,
        "converged": converged,
        "steps": steps,
        "energy": energy,
        "max_gradient": xtb._max_gradient(gradient[free]) / xtb.BOHR,
        "symbols": symbols,
        "positions": (positions * xtb.BOHR).tolist(),
        "gradient": gradient.tolist(),
        "charges": charges,
        "frequencies": None if freqs is None else freqs.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the calculation of the JSON input file in **argv**; return the exit code."""
    return int(xtb.main(argv, runner=run))


if __name__ == "__main__":
    sys.exit(main())
