.. _ASE:

Machine-learned potentials and other ASE calculators
====================================================

CAT can run calculations with any calculator of the
`Atomic Simulation Environment <https://ase-lib.org>`_ (ASE),
*e.g.* machine-learned interatomic potentials such as
`MACE <https://github.com/ACEsuit/mace>`_, through :class:`~CAT.ase.ASEJob`.
It needs ASE and the package of the calculator:

.. code:: bash

    pip install ase  # or: pip install "nlesc-CAT[ase]"
    pip install mace-torch  # for MACE

:class:`~CAT.ase.ASEJob` can be used wherever CAT (or Nano-CAT) accepts a PLAMS job type,
its aliases being ``"ase"`` and ``"asejob"`` (see :ref:`Type Aliases`),
*e.g.* for the ligand and quantum dot optimizations, the bond dissociation energy
and the activation strain analysis.
The calculator is set by the ``calculator`` setting, which is required:

.. code:: yaml

    optional:
        ligand:
            optimize:
                job2: ase
                s2:
                    input:
                        calculator: mace_off
                        calculator_kwargs:
                            model: medium
                            default_dtype: float64

        qd:
            optimize:
                job1: ase
                s1:
                    input:
                        calculator: mace_mp
                        calculator_kwargs:
                            model: medium
                            dispersion: True
                            default_dtype: float64
                job2: null

Like for :class:`~CAT.xtb.TBLiteJob`, the calculations run in a separate process, using the
same Python interpreter as CAT (see :mod:`CAT.ase.runner`), and the geometry optimizations
and frequency analyses are those of :ref:`xTB`, with the same settings.
Their input and output are in the job directory: the input as ``$JN.in`` (JSON),
the log as ``$JN.out``, the results as ``$JN.json`` and the final geometry as ``$JN.xyz``.


Settings
~~~~~~~~

All settings are part of the ``input`` block of the job settings.
Except for those of ``calculator_kwargs``, the keys are insensitive to capitalization and
dashes are equivalent to underscores.
An unknown key makes the job fail.

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    *   - Key
        - Default
        - Description
    *   - ``calculator``
        - (required)
        - The ASE calculator: one of the aliases below, or the import path of a calculator class
          or of a function that returns a calculator, *e.g.* ``ase.calculators.emt.EMT`` or
          ``mace.calculators:mace_mp``.
    *   - ``calculator_kwargs``
        - ``None``
        - The keyword arguments of the calculator.
    *   - ``charge``
        - The sum of the formal charges
        - The total charge.
    *   - ``uhf``
        - ``None``
        - The number of unpaired electrons; ``None`` for the lowest spin state.
    *   - ``geometry``, ``frequencies`` and ``constraints``
        -
        - As for :ref:`xTB`.

The aliases of the calculators are:

=============== =======================================================================
Alias           Calculator
=============== =======================================================================
``emt``         :class:`ase.calculators.emt.EMT`
``lj``          :class:`ase.calculators.lj.LennardJones`
``orca``        :class:`ase.calculators.orca.ORCA`
``tblite``      ``tblite.ase.TBLite``
``mace_mp``     ``mace.calculators.mace_mp`` (MACE-MP, for materials)
``mace_off``    ``mace.calculators.mace_off`` (MACE-OFF, for organic molecules)
=============== =======================================================================

The charge and spin multiplicity are passed to the calculator in ``atoms.info["charge"]``
and ``atoms.info["spin"]`` (as used by *e.g.* FAIR-chem), and in the initial charges and
magnetic moments of the atoms (as used by *e.g.* tblite).
Calculators that take them as keyword arguments, such as ORCA, need them in
``calculator_kwargs``.
Most machine-learned potentials, MACE included, ignore the charge:
they are trained on neutral systems, and their results for the charged ligands of CAT
(*e.g.* carboxylates) should be checked against another method.


API
~~~

.. automodule:: CAT.ase

.. automodule:: CAT.ase.runner
    :members: CALCULATORS, DEFAULTS, get_calculator, run, main
