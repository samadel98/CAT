.. _xTB:

GFN2-xTB with tblite
====================

CAT can run extended tight-binding (xTB) calculations, GFN2-xTB by default,
with `tblite <https://github.com/tblite/tblite>`_.
Unlike the other job types, it needs no separately installed program:
tblite is a Python package with wheels for Linux and macOS.

.. code:: bash

    pip install tblite  # or: pip install "nlesc-CAT[xtb]"

:class:`~CAT.xtb.TBLiteJob` can be used wherever CAT (or Nano-CAT) accepts a PLAMS job type,
its aliases being ``"tblite"``, ``"tblitejob"`` and ``"xtb"`` (see :ref:`Type Aliases`).
For example, for the ligand and quantum dot optimizations, the bond dissociation energy
(:ref:`Bond Dissociation Energy`) and the activation strain analysis:

.. code:: yaml

    optional:
        ligand:
            optimize:
                job2: tblite

        qd:
            optimize:
                job1: tblite
                job2: null

            dissociate:
                core_atom: Cd
                lig_count: 2
                xyn_pre_opt: False  # The pre-optimization requires AMS
                job1: tblite
                job2: tblite

            activation_strain:
                job1: tblite

A job type without settings (``s1`` or ``s2``) runs with the default settings described below,
rather than with the default AMS settings of the other job types.

The calculations run in a separate process, using the same Python interpreter as CAT
(see :mod:`CAT.xtb.runner`).
Their input and output are in the job directory: the input as ``$JN.in`` (JSON),
the log as ``$JN.out``, the results as ``$JN.json`` and the final geometry as ``$JN.xyz``.


Settings
~~~~~~~~

All settings are part of the ``input`` block of the job settings.
For example, GFN1-xTB with the ALPB implicit solvation model of toluene:

.. code:: yaml

    optional:
        qd:
            optimize:
                job1: tblite
                s1:
                    input:
                        method: GFN1-xTB
                        solvent: toluene

The keys are insensitive to capitalization and dashes are equivalent to underscores.
An unknown key makes the job fail.

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    *   - Key
        - Default
        - Description
    *   - ``method``
        - ``GFN2-xTB``
        - The xTB method: ``GFN2-xTB``, ``GFN1-xTB`` or ``IPEA1-xTB``.
    *   - ``charge``
        - See description
        - The total charge. Defaults to the sum of the formal charges of the atoms.
    *   - ``uhf``
        - See description
        - The number of unpaired electrons. Defaults to 0 or, for an odd number of electrons, 1.
    *   - ``solvent``
        - ``null``
        - The name of a solvent (*e.g.* ``water`` or ``toluene``) for implicit solvation.
    *   - ``solvation_model``
        - ``alpb``
        - The implicit solvation model: ``alpb`` or ``gbsa``.
    *   - ``accuracy``
        - ``1.0``
        - The numerical accuracy of the self-consistent charges (SCC); smaller is tighter.
    *   - ``max_iter``
        - ``250``
        - The maximum number of SCC iterations.
    *   - ``electronic_temperature``
        - ``300.0``
        - The electronic temperature in Kelvin.
    *   - ``verbosity``
        - ``0``
        - The amount of output of tblite itself.
    *   - ``geometry.max_iter``
        - ``1000``
        - The maximum number of optimization steps.
    *   - ``geometry.gradient_tol``
        - ``0.001``
        - The optimization converges once the gradient of every free atom is smaller than
          this value, in Hartree/Ångström.
    *   - ``geometry.max_step``
        - ``0.2``
        - The largest displacement of an atom in a single optimization step, in Ångström.
    *   - ``geometry.memory``
        - ``100``
        - The number of previous steps used by the optimizer.
    *   - ``frequencies.step``
        - ``0.0025``
        - The displacement used for computing the Hessian, in Ångström.
    *   - ``constraints.atom``
        - ``[]``
        - The 1-based indices of the atoms that are frozen during geometry optimizations.

The ``task`` key (``singlepoint``, ``geometry`` or ``frequencies``) is set by CAT.


Details
~~~~~~~

Geometry optimizations
    The geometry is optimized in Cartesian coordinates with L-BFGS, without a line search,
    every step being scaled down such that no atom moves more than ``geometry.max_step``
    (like the L-BFGS optimizer of ASE).
    A job fails if the optimization does not converge within ``geometry.max_iter`` steps.

    Like for :class:`~scm.plams.interfaces.adfsuite.ams.AMSJob`, the quantum dot optimization
    (:attr:`optional.qd.optimize`) and the thermochemical part of the bond dissociation
    energy freeze the core atoms and the anchor atoms of the ligands.

Frequencies
    The Hessian is computed with central differences of the gradient,
    which takes six gradient calculations per free atom.
    Translations and rotations are projected out of the Hessian,
    leaving :math:`3n - 6` (:math:`3n - 5` for a linear molecule) frequencies.
    If atoms are frozen, the frequencies are those of the partial Hessian of the free atoms
    (:math:`3n_{free}` frequencies).
    Imaginary frequencies are reported as negative numbers;
    like all frequencies below 100 cm\ :sup:`-1`, they are set to 100 cm\ :sup:`-1` when
    computing thermochemical properties (see :func:`~CAT.thermo_chem.get_thermo`).

Performance
    Ligands take seconds, but quantum dots with hundreds of atoms are expensive:
    for a Cd\ :sub:`68`\ Se\ :sub:`55` core with 26 acetates (305 atoms), as built by CAT,
    a single SCC iteration takes about 6 seconds on 4 cores and the SCC needed over 90 iterations.
    The PyPI wheels of tblite bundle the reference (unoptimized) BLAS and LAPACK libraries;
    the conda-forge packages (``conda install -c conda-forge tblite-python``) link against the
    optimized BLAS and LAPACK of conda-forge instead, which matters most for large systems.
    Raise ``max_iter`` if the SCC does not converge.

    tblite uses OpenMP, its number of threads being set by the ``OMP_NUM_THREADS``
    environment variable.
    Running several tblite calculations at the same time with too many threads in total
    makes them dramatically slower.
    To optimize many ligands or quantum dots, run several jobs at the same time with
    :attr:`optional.parallel_jobs`, which gives each job one thread unless ``OMP_NUM_THREADS``
    is set.


API
~~~

.. automodule:: CAT.xtb

.. automodule:: CAT.xtb.runner
    :members: DEFAULTS, run, main
