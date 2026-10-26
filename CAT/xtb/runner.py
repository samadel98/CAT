"""Run the tblite calculation of a :class:`~CAT.xtb.TBLiteJob`.

The runscript of a :class:`~CAT.xtb.TBLiteJob` runs this file as a script::

    python runner.py JOBNAME.in

It reads the JSON input written by :meth:`TBLiteJob.get_input()<CAT.xtb.TBLiteJob.get_input>`,
writes the results to ``JOBNAME.json`` and the final geometry to ``JOBNAME.xyz``.
It only imports NumPy and tblite, *i.e.* neither CAT nor PLAMS.

Units
-----
The input and output coordinates are in Angstrom, the energy in Hartree,
the gradient in Hartree/Bohr and the frequencies in cm**-1.
The user-facing settings of the geometry optimization and the frequency analysis
(``gradient_tol``, ``max_step`` and ``step``) are in Hartree/Angstrom and Angstrom.

"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

# Python puts the directory of this script first in `sys.path`:
# remove it so that nothing in it can shadow an installed package
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [i for i in sys.path if os.path.abspath(i or os.curdir) != _HERE]

import json
import time
import itertools
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = ["DEFAULTS", "main", "run"]

#: The Bohr radius in Angstrom (CODATA 2018)
BOHR = 0.529177210903

#: The Boltzmann constant in Hartree/Kelvin (CODATA 2018)
KB = 3.166811563e-6

#: Convert the eigenvalues of a mass-weighted Hessian, in Hartree/(Bohr**2 * Dalton),
#: into wavenumbers in cm**-1 (CODATA 2018)
AU_TO_CM = np.sqrt(4.3597447222071e-18 / (BOHR * 1e-10)**2 / 1.66053906660e-27) / (
    2 * np.pi * 2.99792458e10
)

#: The initial guess of the inverse Hessian in Bohr**2/Hartree:
#: the inverse of 70 eV/Angstrom**2, the default of the L-BFGS optimizer of ASE
H0 = 1 / (70 / 27.211386245988 * BOHR**2)

#: The default input settings.
#: Their keys, including those of the nested dictionaries, are the only valid input keys.
DEFAULTS: Mapping[str, Any] = {
    "task": "singlepoint",
    "method": "GFN2-xTB",
    "charge": 0,
    "uhf": None,
    "accuracy": 1.0,
    "max_iter": 250,
    "electronic_temperature": 300.0,
    "solvent": None,
    "solvation_model": "alpb",
    "verbosity": 0,
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
SOLVATION_MODELS = frozenset({"alpb", "gbsa"})

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


def parse_input(inp: Mapping[str, Any], defaults: Mapping[str, Any] = DEFAULTS,
                prefix: str = "") -> dict[str, Any]:
    """Return **defaults** updated with **inp**, raising a :exc:`KeyError` for unknown keys.

    Keys are case insensitive and dashes are equivalent to underscores.

    """
    ret = dict(defaults)
    for key, value in inp.items():
        k = str(key).lower().replace("-", "_")
        if k not in defaults:
            raise ValueError(f"Unknown TBLiteJob setting: 'input.{prefix}{key}'")
        elif isinstance(defaults[k], Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"'input.{prefix}{key}' expects a dictionary")
            ret[k] = parse_input(value, defaults[k], prefix=f"{prefix}{k}.")
        else:
            ret[k] = value

    task = ret.get("task")
    if not prefix and str(task).lower() not in TASKS:
        raise ValueError(f"Invalid TBLiteJob task: {task!r}; expected one of {sorted(TASKS)}")
    model = ret.get("solvation_model")
    if not prefix and str(model).lower() not in SOLVATION_MODELS:
        raise ValueError(f"Invalid TBLiteJob solvation model: {model!r}; "
                         f"expected one of {sorted(SOLVATION_MODELS)}")
    return ret


class Engine:
    """Compute the xTB energy and gradient of a molecule with tblite, in atomic units.

    Each calculation starts from the wavefunction of the last one that was kept
    (see the **keep** parameter of :meth:`__call__`).

    """

    def __init__(self, numbers: np.ndarray, positions: np.ndarray,
                 inp: Mapping[str, Any]) -> None:
        from tblite.interface import Calculator

        self.charge = inp["charge"]
        self.uhf = inp["uhf"]
        if self.uhf is None:  # The lowest spin state for this number of electrons
            self.uhf = int(round(numbers.sum() - self.charge)) % 2

        self.calc = Calculator(
            inp["method"], numbers, positions, charge=self.charge, uhf=self.uhf
        )
        self.calc.set("verbosity", inp["verbosity"])
        self.calc.set("accuracy", inp["accuracy"])
        self.calc.set("max-iter", inp["max_iter"])
        self.calc.set("temperature", inp["electronic_temperature"] * KB)
        if inp["solvent"] is not None:
            model = inp["solvation_model"].lower()
            self.calc.add(f"{model}-solvation", inp["solvent"])

        self.result: Any = None
        self.count = 0

    def __call__(self, positions: np.ndarray, keep: bool = True) -> tuple[float, np.ndarray]:
        """Return the energy (Hartree) and gradient (Hartree/Bohr) at **positions** (Bohr).

        The wavefunction is kept as the initial guess of the next calculation if **keep** is true.

        """
        self.calc.update(positions)
        result = self.calc.singlepoint(self.result, copy=not keep)
        if keep:
            self.result = result
        self.count += 1
        return float(result.get("energy")), np.array(result.get("gradient"))


def _max_gradient(gradient: np.ndarray) -> float:
    """Return the largest atomic gradient norm."""
    return float(np.linalg.norm(gradient, axis=1).max(initial=0.0))


def optimize(
    engine: Engine,
    positions: np.ndarray,
    free: np.ndarray,
    max_iter: int = 1000,
    gradient_tol: float = 1e-3,
    max_step: float = 0.2,
    memory: int = 100,
    log: Log = _print,
) -> tuple[np.ndarray, float, np.ndarray, bool, int]:
    """Minimize the energy with L-BFGS, moving only the atoms in **free**.

    Following the L-BFGS optimizer of ASE, there is no line search,
    but every step is scaled down such that no atom moves more than **max_step**.
    The optimization converges once no free atom has a gradient norm
    larger than **gradient_tol**.

    Parameters
    ----------
    engine : :class:`Engine`
        The calculator.
    positions : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n, 3)`
        The initial coordinates in Bohr.
    free : :class:`np.ndarray[np.bool_] <numpy.ndarray>`, shape :math:`(n,)`
        Whether each atom is allowed to move.
    max_iter : :class:`int`
        The maximum number of steps.
    gradient_tol : :class:`float`
        The convergence threshold in Hartree/Angstrom.
    max_step : :class:`float`
        The largest displacement of an atom in a single step, in Angstrom.
    memory : :class:`int`
        The number of previous steps used for approximating the Hessian.
    log : :class:`Callable[[str], None] <collections.abc.Callable>`
        A function for logging the progress.

    Returns
    -------
    :class:`tuple`
        The optimized coordinates (Bohr), energy (Hartree) and gradient (Hartree/Bohr),
        whether the optimization converged and the number of steps.

    """
    step_max = max_step / BOHR  # Angstrom -> Bohr

    s_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rho_list: list[float] = []

    x = np.array(positions, dtype=np.float64)
    energy, gradient = engine(x)
    log(f"{'step':>6} {'energy / Hartree':>20} {'max gradient / Hartree/Angstrom':>32}")
    for i in range(max_iter + 1):
        g = gradient[free].ravel()
        g_max = _max_gradient(gradient[free]) / BOHR
        log(f"{i:6d} {energy:20.10f} {g_max:32.6e}")
        if g_max <= gradient_tol:
            return x, energy, gradient, True, i
        elif i == max_iter:
            break

        # The two-loop recursion of L-BFGS
        q = g.copy()
        alpha = np.empty(len(s_list))
        for j in reversed(range(len(s_list))):
            alpha[j] = rho_list[j] * (s_list[j] @ q)
            q -= alpha[j] * y_list[j]
        z = H0 * q
        for j in range(len(s_list)):
            beta = rho_list[j] * (y_list[j] @ z)
            z += s_list[j] * (alpha[j] - beta)
        step = -z
        if step @ g >= 0:  # Not a descent direction: restart from steepest descent
            s_list.clear()
            y_list.clear()
            rho_list.clear()
            step = -H0 * g

        step_3d = step.reshape(-1, 3)
        longest = np.linalg.norm(step_3d, axis=1).max()
        if longest > step_max:
            step_3d *= step_max / longest

        x_new = x.copy()
        x_new[free] += step_3d
        energy_new, gradient_new = engine(x_new)

        # Only keep steps that satisfy the curvature condition,
        # so that the approximate Hessian remains positive definite
        y = gradient_new[free].ravel() - g
        s = step_3d.ravel()
        sy = s @ y
        if sy > 1e-12:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1 / sy)
            if len(s_list) > memory:
                del s_list[0], y_list[0], rho_list[0]
        x, energy, gradient = x_new, energy_new, gradient_new
    return x, energy, gradient, False, max_iter


def hessian(engine: Engine, positions: np.ndarray, free: np.ndarray,
            step: float = 0.0025) -> np.ndarray:
    """Return the Hessian of the free atoms in Hartree/Bohr**2.

    The Hessian is computed with central differences of the gradient,
    displacing each Cartesian coordinate of a free atom by **step** Angstrom.

    """
    h = step / BOHR
    idx = np.flatnonzero(free)
    ret = np.empty((3 * len(idx), 3 * len(idx)))
    x = np.array(positions, dtype=np.float64)
    for k, (i, j) in enumerate(itertools.product(idx, range(3))):
        x[i, j] += h
        _, gradient_plus = engine(x, keep=False)
        x[i, j] -= 2 * h
        _, gradient_minus = engine(x, keep=False)
        x[i, j] += h
        ret[k] = (gradient_plus[idx] - gradient_minus[idx]).ravel() / (2 * h)
    return (ret + ret.T) / 2


def vibrational_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis of the mass-weighted coordinates without translations and rotations.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(3n, 3n - k)`
        The basis vectors, with :math:`k = 6` (:math:`k = 5` for a linear molecule).

    """  # noqa: E501
    sqrt_m = np.sqrt(masses)[:, None]
    r = positions - masses @ positions / masses.sum()

    tr = np.zeros((6, len(masses), 3))
    for k, axis in enumerate(np.eye(3)):
        tr[k, :, k] = sqrt_m[:, 0]
        tr[3 + k] = np.cross(axis, r) * sqrt_m
    u, s, _ = np.linalg.svd(tr.reshape(6, -1).T, full_matrices=True)
    rank = np.count_nonzero(s > 1e-6 * s.max())
    return u[:, rank:]


def frequencies(hess: np.ndarray, masses: np.ndarray, positions: np.ndarray | None = None,
                ) -> np.ndarray:
    """Return the vibrational frequencies in cm**-1, imaginary ones as negative numbers.

    Translations and rotations are projected out of the Hessian if **positions**
    (Bohr) are given, *i.e.* if all atoms are free.

    """
    m = np.repeat(masses, 3)
    hess_mw = hess / np.sqrt(np.outer(m, m))
    if positions is not None:
        basis = vibrational_basis(positions, masses)
        hess_mw = basis.T @ hess_mw @ basis
    eigval = np.linalg.eigvalsh(hess_mw)
    return np.sign(eigval) * np.sqrt(np.abs(eigval)) * AU_TO_CM


def _get_free(atoms: int | list[int], count: int) -> np.ndarray:
    """Convert the 1-based indices of the frozen atoms into a mask of the free atoms."""
    idx = np.atleast_1d(np.asarray(atoms, dtype=np.int64))
    if idx.size and (idx.min() < 1 or idx.max() > count):
        raise IndexError(f"'input.constraints.atom' expects 1-based atomic indices "
                         f"between 1 and {count}; observed values: {idx.tolist()}")
    ret = np.ones(count, dtype=np.bool_)
    ret[idx - 1] = False
    return ret


def run(data: Mapping[str, Any], log: Log = _print) -> dict[str, Any]:
    """Run the calculation described by the JSON input **data** and return its results.

    **data** consists of the ``"input"`` settings (see :data:`DEFAULTS`) and the ``"atoms"``,
    a list with the symbol, atomic number, mass and Cartesian coordinates (Angstrom)
    of each atom.

    """
    inp = parse_input(data.get("input", {}))
    task = inp["task"].lower()
    symbols = [at[0] for at in data["atoms"]]
    numbers = np.array([at[1] for at in data["atoms"]], dtype=np.int64)
    masses = np.array([at[2] for at in data["atoms"]], dtype=np.float64)
    positions = np.array([at[3:6] for at in data["atoms"]], dtype=np.float64) / BOHR
    free = _get_free(inp["constraints"]["atom"], len(numbers))

    start = time.perf_counter()
    engine = Engine(numbers, positions, inp)
    log(f"tblite {inp['method']} {task}: {len(numbers)} atoms ({np.count_nonzero(~free)} frozen), "
        f"charge {engine.charge}, {engine.uhf} unpaired electron(s)")

    converged = True
    steps = 0
    if task == "geometry":
        positions, energy, gradient, converged, steps = optimize(
            engine, positions, free, log=log, **inp["geometry"]
        )
    else:
        energy, gradient = engine(positions)
    charges = np.array(engine.result.get("charges"))

    freqs = None
    if task == "frequencies":
        hess = hessian(engine, positions, free, step=inp["frequencies"]["step"])
        freqs = frequencies(hess, masses[free], positions if free.all() else None)
        n_imag = np.count_nonzero(freqs < 0)
        log(f"{len(freqs)} vibrational frequencies, {n_imag} imaginary")

    log(f"Final energy: {energy:.10f} Hartree "
        f"({engine.count} calculations in {time.perf_counter() - start:.2f} s)")
    return {
        "task": task,
        "method": inp["method"],
        "charge": engine.charge,
        "uhf": engine.uhf,
        "converged": converged,
        "steps": steps,
        "energy": energy,
        "max_gradient": _max_gradient(gradient[free]) / BOHR,
        "symbols": symbols,
        "positions": (positions * BOHR).tolist(),
        "gradient": gradient.tolist(),
        "charges": charges.tolist(),
        "frequencies": None if freqs is None else freqs.tolist(),
    }


def write_xyz(filename: str, symbols: list[str], positions: list[list[float]],
              comment: str = "") -> None:
    """Write a single geometry to an .xyz file."""
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for symbol, (x, y, z) in zip(symbols, positions):
            f.write(f"{symbol:<3} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the calculation of the JSON input file in **argv**; return the exit code."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python runner.py JOBNAME.in", file=sys.stderr)
        return 2
    root = os.path.splitext(args[0])[0]

    try:
        with open(args[0]) as f:
            data = json.load(f)
        ret = run(data)
    except Exception as ex:
        # CAT reports the first line of stderr as the reason why a job failed
        print(f"{ex.__class__.__name__}: {ex}", file=sys.stderr)
        traceback.print_exc()
        return 1

    with open(f"{root}.json", "w") as f:
        json.dump(ret, f, indent=4)
    comment = f"energy = {ret['energy']!r} Hartree"
    write_xyz(f"{root}.xyz", ret["symbols"], ret["positions"], comment)

    if not ret["converged"]:
        print(f"The geometry optimization did not converge in {ret['steps']} steps; "
              f"the largest atomic gradient is {ret['max_gradient']:.3e} Hartree/Angstrom",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
