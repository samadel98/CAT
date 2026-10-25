.. _import_qd:

Importing Quantum Dots
======================

Import pre-built quantum dots rather than constructing them from scratch,
*e.g.* experimental structures, MD snapshots or quantum dots built with other tools.
The imported quantum dots can then be used in the usual quantum dot workflows
(:ref:`optimization <Optional>`, activation strain analyses, ligand dissociation, *etc.*).
Ligand bulkiness calculations are not supported, as they need the separate core and ligand.

All ligands must be identical and are identified by matching the connectivity of
:attr:`ligand_smiles` against the quantum dot; all remaining atoms are treated as the core.
If the file does not provide bonds (*e.g.* an .xyz file), they are guessed first.
After the import:

* the atoms are reordered: the core first, followed by one ligand at a time
  (in the atom order of :attr:`ligand_smiles`);
* the bonds and formal charges of the ligands are taken from :attr:`ligand_smiles`,
  while core atoms get CAT's default charges (*e.g.* +2 for Cd and -2 for Se);
* symmetry-equivalent anchors (*e.g.* the two oxygens of a carboxylate) are assigned
  such that the anchor is the atom closest to the core;
* in the results, the ``"core anchor"`` index level holds the name of the imported file,
  as there are no core dummy atoms to refer to.


Default Settings
~~~~~~~~~~~~~~~~

.. code:: yaml

    input_qd:
        - Cd68Se55_ethoxide.xyz:
            ligand_smiles: '[O-]CC'
            ligand_anchor: '[O-]'
            guess_bonds: False  # Bonds are always guessed if the file has none

|

Arguments
~~~~~~~~~

.. attribute:: ligand_smiles

    :Parameter:     * **Type** - :class:`str`
                    * **Default value** – ``None``

    A SMILES string representing the ligand, including its charge (*e.g.* ``"CCC(=O)[O-]"``).
    The provided SMILES string will be used for identifying the core and all ligands.

    .. warning::
        This argument has no value be default and thus *must* be provided by the user.


.. attribute:: ligand_anchor

    :Parameter:     * **Type** - :class:`str`
                    * **Default value** – ``None``

    A SMILES (or SMARTS) string representing the anchor functional group of the ligand.
    If the provided string consists of multiple atoms
    (*e.g.* a carboxylate: ``"[O-]C=O"``), then the first atom will be treated as anchor (``"[O-]"``).

    .. warning::
        This argument has no value be default and thus *must* be provided by the user.
