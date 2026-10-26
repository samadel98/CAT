"""A module with various data-types used throughout **Data-CAT**.

Index
-----
.. currentmodule:: dataCAT.dtype
.. autosummary::
    ATOMS_DTYPE
    BONDS_DTYPE
    ATOM_COUNT_DTYPE
    BOND_COUNT_DTYPE
    LIG_IDX_DTYPE
    QD_IDX_DTYPE
    BACKUP_IDX_DTYPE

    DT_DTYPE
    VERSION_DTYPE
    INDEX_DTYPE
    MSG_DTYPE

API
---
.. autodata:: ATOMS_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Most field names are based on to their, identically named, counterpart as produced by
    :func:`readpdb()<scm.plams.interfaces.molecule.rdkit.readpdb>`,
    the data in question being stored in the
    :class:`Atom.properties.pdb_info<scm.plams.mol.atom.Atom>` block.

    There are six exception to this general rule:

    * ``x``, ``y`` & ``z``: Based on :class:`Atom.x<scm.plams.mol.atom.Atom>`,
      :class:`Atom.y<scm.plams.mol.atom.Atom>` and :class:`Atom.z<scm.plams.mol.atom.Atom>`.
    * ``symbol``: Based on :class:`Atom.symbol<scm.plams.mol.atom.Atom>`.
    * ``charge``: Based on :class:`Atom.properties.charge<scm.plams.mol.atom.Atom>`.
    * ``charge_float``: Based on :class:`Atom.properties.charge_float<scm.plams.mol.atom.Atom>`.

    .. code:: python

        >>> from dataCAT.dtype import ATOMS_DTYPE

        >>> print(repr(ATOMS_DTYPE))  # doctest: +NORMALIZE_WHITESPACE
        dtype([('IsHeteroAtom', '?'),
               ('SerialNumber', '<i2'),
               ('Name', 'S4'),
               ('ResidueName', 'S3'),
               ('ChainId', 'S1'),
               ('ResidueNumber', '<i2'),
               ('x', '<f4'),
               ('y', '<f4'),
               ('z', '<f4'),
               ('Occupancy', '<f4'),
               ('TempFactor', '<f4'),
               ('symbol', 'S4'),
               ('charge', 'i1'),
               ('charge_float', '<f8')])

.. autodata:: BONDS_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Field names are based on to their, identically named,
    counterpart in :class:`plams.Bond<scm.plams.mol.bond.Bond>`.

    .. code:: python

        >>> from dataCAT.dtype import BONDS_DTYPE

        >>> print(repr(BONDS_DTYPE))  # doctest: +NORMALIZE_WHITESPACE
        dtype([('atom1', '<i4'),
               ('atom2', '<i4'),
               ('order', 'i1')])

.. autodata:: ATOM_COUNT_DTYPE
    :annotation: : numpy.dtype = ...

    .

    .. code:: python

        >>> from dataCAT.dtype import ATOM_COUNT_DTYPE

        >>> print(repr(ATOM_COUNT_DTYPE))
        dtype('int32')

.. autodata:: BOND_COUNT_DTYPE
    :annotation: : numpy.dtype = ...

    .

    .. code:: python

        >>> from dataCAT.dtype import BOND_COUNT_DTYPE

        >>> print(repr(BOND_COUNT_DTYPE))
        dtype('int32')

.. autodata:: LIG_IDX_DTYPE
    :annotation: : numpy.dtype = ...

    .

    .. code:: python

        >>> import h5py
        >>> from dataCAT.dtype import LIG_IDX_DTYPE

        >>> print(repr(LIG_IDX_DTYPE))  # doctest: +NORMALIZE_WHITESPACE
        dtype([('ligand', 'O'),
               ('ligand anchor', 'O')])

        >>> h5py.check_string_dtype(LIG_IDX_DTYPE.fields['ligand'][0])
        string_info(encoding='ascii', length=None)

        >>> h5py.check_string_dtype(LIG_IDX_DTYPE.fields['ligand anchor'][0])
        string_info(encoding='ascii', length=None)

.. autodata:: QD_IDX_DTYPE
    :annotation: : numpy.dtype = ...

    .

    .. code:: python

        >>> import h5py
        >>> from dataCAT.dtype import QD_IDX_DTYPE

        >>> print(repr(QD_IDX_DTYPE))  # doctest: +NORMALIZE_WHITESPACE
        dtype([('core', 'O'),
               ('core anchor', 'O'),
               ('ligand', 'O'),
               ('ligand anchor', 'O')])

        >>> h5py.check_string_dtype(QD_IDX_DTYPE.fields['core'][0])
        string_info(encoding='ascii', length=None)

        >>> h5py.check_string_dtype(QD_IDX_DTYPE.fields['core anchor'][0])
        string_info(encoding='ascii', length=None)

        >>> h5py.check_string_dtype(QD_IDX_DTYPE.fields['ligand'][0])
        string_info(encoding='ascii', length=None)

        >>> h5py.check_string_dtype(QD_IDX_DTYPE.fields['ligand anchor'][0])
        string_info(encoding='ascii', length=None)

.. autodata:: BACKUP_IDX_DTYPE
    :annotation: : numpy.dtype = ...

    .

    .. code:: python

        >>> from dataCAT.dtype import BACKUP_IDX_DTYPE

        >>> print(repr(BACKUP_IDX_DTYPE))
        dtype('int32')

.. autodata:: DT_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Field names are based on their, identically named, counterpart in
    the :class:`~datetime.datetime` class.

    .. code:: python

        >>> from dataCAT.dtype import DT_DTYPE

        >>> print(repr(DT_DTYPE))  # doctest: +NORMALIZE_WHITESPACE
        dtype([('year', '<i2'),
               ('month', 'i1'),
               ('day', 'i1'),
               ('hour', 'i1'),
               ('minute', 'i1'),
               ('second', 'i1'),
               ('microsecond', '<i4')])

.. autodata:: VERSION_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Field names are based on their, identically named, counterpart in
    the :class:`nanoutils.VersionInfo` namedtuple.

    .. code:: python

        >>> from dataCAT.dtype import VERSION_DTYPE

        >>> print(repr(VERSION_DTYPE))  # doctest: +NORMALIZE_WHITESPACE
        dtype([('major', 'i1'),
               ('minor', 'i1'),
               ('micro', 'i1')])

.. autodata:: INDEX_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Used for representing a ragged array of 32-bit integers.

    .. code:: python

        >>> import h5py
        >>> from dataCAT.dtype import INDEX_DTYPE

        >>> print(repr(INDEX_DTYPE))
        dtype('O')

        >>> h5py.check_vlen_dtype(INDEX_DTYPE)
        dtype('int32')

.. autodata:: MSG_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Used for representing variable-length ASCII strings.

    .. code:: python

        >>> import h5py
        >>> from dataCAT.dtype import MSG_DTYPE

        >>> print(repr(MSG_DTYPE))
        dtype('O')

        >>> h5py.check_string_dtype(MSG_DTYPE)
        string_info(encoding='ascii', length=None)

.. autodata:: FORMULA_DTYPE
    :annotation: : numpy.dtype = ...

    .

    Used for representing variable-length ASCII strings.

    .. code:: python

        >>> import h5py
        >>> from dataCAT.dtype import FORMULA_DTYPE

        >>> print(repr(FORMULA_DTYPE))
        dtype('O')

        >>> h5py.check_string_dtype(FORMULA_DTYPE)
        string_info(encoding='ascii', length=None)


.. autodata:: LIG_COUNT_DTYPE
    :annotation: : numpy.dtype = ...

    .

    .. code:: python

        >>> from dataCAT.dtype import LIG_COUNT_DTYPE

        >>> print(repr(LIG_COUNT_DTYPE))
        dtype('int32')

"""

import h5py
import numpy as np

__all__ = [
    'ATOMS_DTYPE', 'BONDS_DTYPE', 'ATOM_COUNT_DTYPE', 'BOND_COUNT_DTYPE',

    'DT_DTYPE', 'VERSION_DTYPE', 'INDEX_DTYPE', 'MSG_DTYPE',

    'LIG_IDX_DTYPE', 'QD_IDX_DTYPE', 'BACKUP_IDX_DTYPE',

    'FORMULA_DTYPE', 'LIG_COUNT_DTYPE'
]

_ATOMS_MAPPING = {
    'IsHeteroAtom': 'bool',
    'SerialNumber': 'int16',
    'Name': 'S4',
    'ResidueName': 'S3',
    'ChainId': 'S1',
    'ResidueNumber': 'int16',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32',
    'Occupancy': 'float32',
    'TempFactor': 'float32',
    'symbol': 'S4',
    'charge': 'int8',
    'charge_float': 'float64'
}
#: The datatype of :attr:`PDBContainer.atoms<dataCAT.PDBContainer.atoms>`
ATOMS_DTYPE = np.dtype(list(_ATOMS_MAPPING.items()))


_BONDS_MAPPING = {
    'atom1': 'int32',
    'atom2': 'int32',
    'order': 'int8'
}
#: The datatype of :attr:`PDBContainer.bonds<dataCAT.PDBContainer.bonds>`
BONDS_DTYPE = np.dtype(list(_BONDS_MAPPING.items()))

#: The datatype of :attr:`PDBContainer.atom_count<dataCAT.PDBContainer.atom_count>`
ATOM_COUNT_DTYPE = np.dtype('int32')

#: The datatype of :attr:`PDBContainer.bond_count<dataCAT.PDBContainer.bond_count>`
BOND_COUNT_DTYPE = np.dtype('int32')


_DT_MAPPING = {
    'year': 'int16',
    'month': 'int8',
    'day': 'int8',
    'hour': 'int8',
    'minute': 'int8',
    'second': 'int8',
    'microsecond': 'int32'
}
#: The datatype of the ``"date"`` dataset created by :func:`~dataCAT.create_hdf5_log`
DT_DTYPE = np.dtype(list(_DT_MAPPING.items()))


_VERSION_MAPPING = {
    'major': 'int8',
    'minor': 'int8',
    'micro': 'int8'
}
#: The datatype of the ``"version"`` dataset created by :func:`~dataCAT.create_hdf5_log`
VERSION_DTYPE = np.dtype(list(_VERSION_MAPPING.items()))

#: The datatype of the ``"index"`` dataset created by :func:`~dataCAT.create_hdf5_log`
INDEX_DTYPE = h5py.vlen_dtype(np.dtype('int32'))

#: The datatype of the ``"message"`` dataset created by :func:`~dataCAT.create_hdf5_log`
MSG_DTYPE = h5py.string_dtype(encoding='ascii')

_LIG_IDX_MAPPING = {
    'ligand': h5py.string_dtype(encoding='ascii'),
    'ligand anchor': h5py.string_dtype(encoding='ascii')
}
#: The datatype of :attr:`PDBContainer.index<dataCAT.PDBContainer.index>`
#: as used by the ligand database
LIG_IDX_DTYPE = np.dtype(list(_LIG_IDX_MAPPING.items()))


_QD_IDX_MAPPING = {
    'core': h5py.string_dtype(encoding='ascii'),
    'core anchor': h5py.string_dtype(encoding='ascii'),
    'ligand': h5py.string_dtype(encoding='ascii'),
    'ligand anchor': h5py.string_dtype(encoding='ascii')
}
#: The datatype of :attr:`PDBContainer.index<dataCAT.PDBContainer.index>`
#: as used by the QD database
QD_IDX_DTYPE = np.dtype(list(_QD_IDX_MAPPING.items()))

#: The default datatype of :attr:`PDBContainer.index<dataCAT.PDBContainer.index>`
BACKUP_IDX_DTYPE = np.dtype('int32')

#: The datatype of the ``"/ligand/properties/formula"`` dataset.
FORMULA_DTYPE = h5py.string_dtype(encoding='ascii')

#: The datatype of the ``"/qd/properties/ligand count"`` dataset.
LIG_COUNT_DTYPE = np.dtype('int32')
