"""A module for constructing array-representations of .pdb files.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    PDBContainer
    PDBContainer.atoms
    PDBContainer.bonds
    PDBContainer.atom_count
    PDBContainer.bond_count
    PDBContainer.scale

.. autosummary::
    PDBContainer.__init__
    PDBContainer.__getitem__
    PDBContainer.__len__
    PDBContainer.keys
    PDBContainer.values
    PDBContainer.items
    PDBContainer.concatenate

.. autosummary::
    PDBContainer.from_molecules
    PDBContainer.to_molecules
    PDBContainer.to_rdkit
    PDBContainer.create_hdf5_group
    PDBContainer.validate_hdf5
    PDBContainer.from_hdf5
    PDBContainer.to_hdf5

.. autosummary::
    PDBContainer.intersection
    PDBContainer.difference
    PDBContainer.symmetric_difference
    PDBContainer.union


API
---
.. autoclass:: PDBContainer
    :members: atoms, bonds, atom_count, bond_count, scale, __init__

API: Miscellaneous Methods
--------------------------
.. automethod:: PDBContainer.__getitem__
.. automethod:: PDBContainer.__len__
.. automethod:: PDBContainer.keys
.. automethod:: PDBContainer.values
.. automethod:: PDBContainer.items
.. automethod:: PDBContainer.concatenate

API: Object Interconversion
---------------------------
.. automethod:: PDBContainer.from_molecules
.. automethod:: PDBContainer.to_molecules
.. automethod:: PDBContainer.to_rdkit
.. automethod:: PDBContainer.create_hdf5_group
.. automethod:: PDBContainer.validate_hdf5
.. automethod:: PDBContainer.from_hdf5
.. automethod:: PDBContainer.to_hdf5

API: Set Operations
-------------------
.. automethod:: PDBContainer.intersection
.. automethod:: PDBContainer.difference
.. automethod:: PDBContainer.symmetric_difference
.. automethod:: PDBContainer.union

"""  # noqa: E501

from __future__ import annotations

import textwrap
from types import MappingProxyType
from itertools import repeat, chain
from typing import (
    TypeVar, Any, Literal, SupportsIndex,
    TypedDict, overload, ClassVar, TYPE_CHECKING
)
from collections.abc import Iterable, Sequence, Mapping, Generator

import h5py
import numpy as np
from scm.plams import Molecule, Atom, Bond
from rdkit import Chem, Geometry
from assertionlib import assertion

from .dtype import ATOMS_DTYPE, BONDS_DTYPE, ATOM_COUNT_DTYPE, BOND_COUNT_DTYPE, BACKUP_IDX_DTYPE
from .functions import int_to_slice, if_exception

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

__all__ = ['PDBContainer']

ST = TypeVar('ST', bound='PDBContainer')

#: The value of the ``copy`` parameter of :func:`numpy.array` for only copying when needed.
_COPY_IF_NEEDED = None if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else False

_AtomTuple = tuple[
    bool,  # IsHeteroAtom
    int,  # SerialNumber
    str,  # Name
    str,  # ResidueName
    str,  # ChainId
    int,  # ResidueNumber
    float,  # x
    float,  # y
    float,  # z
    float,  # Occupancy
    float,  # TempFactor
    str,  # symbol
    int,  # charge
    float  # charge_float
]

_BondTuple = tuple[
    int,  # atom1
    int,  # atom2
    int,  # order
]

_ReduceTuple = tuple[
    np.recarray,  # atoms
    np.recarray,  # bonds
    np.ndarray,  # atom_count
    np.ndarray,  # bond_count
    np.recarray,  # scale
    Literal[False]  # validate
]

_Coords = tuple[float, float, float]

_ResInfo = tuple[
    str,  # Name,
    int,  # SerialNumber,
    str,  # AltLoc
    str,  # ResidueName
    int,  # ResidueNumber
    str,  # ChainId
    str,  # InsertionCode
    float,  # Occupancy
    float,  # TempFactor
    bool,  # IsHeteroAtom
]


class _PDBInfo(TypedDict):
    IsHeteroAtom: bool
    SerialNumber: int
    Name: str
    ResidueName: str
    ChainId: str
    ResidueNumber: int
    Occupancy: float
    TempFactor: float


class _Properties(TypedDict):
    charge: int
    charge_float: float
    pdb_info: _PDBInfo


def _get_atom_info(at: Atom, i: int) -> _AtomTuple:
    """Helper function for :meth:`PDBContainer.from_molecules`: create a tuple representing a single :attr:`PDBContainer.atoms` row."""  # noqa: E501
    prop = at.properties
    symbol = at.symbol
    charge = prop.get('charge', 0)

    pdb = prop.get('pdb_info', {})
    return (
        pdb.get('IsHeteroAtom', False),
        pdb.get('SerialNumber', i),
        pdb.get('Name', symbol),
        pdb.get('ResidueName', 'LIG'),
        pdb.get('ChainId', 'A'),
        pdb.get('ResidueNumber', 1),
        *at.coords,
        pdb.get('Occupancy', 1.0),
        pdb.get('TempFactor', 0.0),
        symbol,
        charge,
        prop.get('charge_float', charge)
    )


def _get_bond_info(mol: Molecule) -> list[_BondTuple]:
    """Helper function for :meth:`PDBContainer.from_molecules`: create a tuple representing a single :attr:`PDBContainer.bonds` row.

    Note that the atomic indices are 1-based.
    """  # noqa: E501
    mol.set_atoms_id(start=1)
    ret = [(b.atom1.id, b.atom2.id, b.order) for b in mol.bonds]
    mol.unset_atoms_id()
    return ret


def _iter_rec_plams(
    atom_array: np.ndarray,
) -> Generator[tuple[_Properties, _Coords, str], None, None]:
    """Helper function for :func:`_rec_to_mol`: create an iterator yielding atom properties and attributes."""  # noqa: E501
    for ar in atom_array:
        IsHeteroAtom, SerialNumber, Name, ResidueName, ChainId, ResidueNumber, x, y, z, Occupancy, TempFactor, symbol, charge, charge_float = ar.item()  # noqa: E501
        _pdb_info = {
            'IsHeteroAtom': IsHeteroAtom,
            'SerialNumber': SerialNumber,
            'Name': Name.decode(),
            'ResidueName': ResidueName.decode(),
            'ChainId': ChainId.decode(),
            'ResidueNumber': ResidueNumber,
            'Occupancy': Occupancy,
            'TempFactor': TempFactor
        }

        properties = {
            'charge': charge,
            'charge_float': charge_float,
            'pdb_info': _pdb_info
        }
        yield properties, (x, y, z), symbol.decode()  # type: ignore


def _rec_to_plams(
    atom_array: np.recarray,
    bond_array: np.recarray,
    atom_len: None | int = None,
    bond_len: None | int = None,
    mol: None | Molecule = None,
) -> Molecule:
    """Helper function for :meth:`PDBContainer.to_molecules`: update/create a single plams molecule from the passed **atom_array** and **bond_array**."""  # noqa: E501
    if mol is None:
        ret = Molecule()
        for _ in range(len(atom_array[:atom_len])):
            ret.add_atom(Atom(mol=ret))
    else:
        ret = mol

    iterator = _iter_rec_plams(atom_array[:atom_len])
    for atom, (properties, coords, symbol) in zip(ret, iterator):
        atom.coords = coords
        atom.symbol = symbol
        atom.properties.update(properties)

    if ret.bonds:
        ret.delete_all_bonds()
    for i, j, order in bond_array[:bond_len]:
        bond = Bond(atom1=ret[i], atom2=ret[j], order=order, mol=ret)
        ret.add_bond(bond)
    return ret


def _iter_rec_rdkit(
    atom_array: np.ndarray,
) -> Generator[tuple[str, int, _ResInfo], None, None]:
    """Helper function for :func:`_rec_to_mol`: create an iterator yielding atom properties and attributes."""  # noqa: E501
    for ar in atom_array:
        IsHeteroAtom, SerialNumber, Name, ResidueName, ChainId, ResidueNumber, x, y, z, Occupancy, TempFactor, symbol, charge, charge_float = ar.item()  # noqa: E501
        res_info = (
            Name,
            SerialNumber,
            "",
            ResidueName,
            ResidueNumber,
            ChainId,
            "",
            Occupancy,
            TempFactor,
            IsHeteroAtom,
        )
        yield symbol, charge, res_info


def _rec_to_rdkit(
    atom_array: np.recarray,
    bond_array: np.recarray,
    atom_len: None | int = None,
    bond_len: None | int = None,
    sanitize: bool = True,
) -> Chem.Mol:
    """Helper function for :meth:`PDBContainer.to_rdkit`: create a single rdkit molecule from the passed **atom_array** and **bond_array**."""  # noqa: E501
    edit_mol = Chem.EditableMol(Chem.Mol())

    iterator1 = _iter_rec_rdkit(atom_array[:atom_len])
    for symbol, charge, res_info in iterator1:
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(*res_info))
        edit_mol.AddAtom(atom)

    for void in bond_array[:bond_len]:
        i, j, order = void.item()
        edit_mol.AddBond(i - 1, j - 1, Chem.BondType(order))

    mol = edit_mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)

    conf = Chem.Conformer()
    iterator2 = (void.item()[6:9] for void in atom_array[:atom_len])
    for i, xyz in enumerate(iterator2):
        conf.SetAtomPosition(i, Geometry.Point3D(*xyz))
    mol.AddConformer(conf)
    return mol


IndexLike = None | SupportsIndex | Sequence[int] | slice | np.ndarray
Hdf5Mode = Literal['append', 'update']
_AttrName = Literal['atoms', 'bonds', 'atom_count', 'bond_count', 'scale']

#: The docstring to-be assigned by :meth:`PDBContainer.create_hdf5_group`.
HDF5_DOCSTRING = """A Group of datasets representing :class:`{cls_name}`."""


class PDBContainer:
    """An (immutable) class for holding array-like representions of a set of .pdb files.

    The :class:`PDBContainer` class serves as an (intermediate) container
    for storing .pdb files in the hdf5 format,
    thus facilitating the storage and interconversion
    between PLAMS molecules and the :mod:`h5py` interface.

    The methods implemented in this class can roughly be divided into three categories:

    * Molecule-interconversion: :meth:`~PDBContainer.to_molecules`,
      :meth:`~PDBContainer.from_molecules` & :meth:`~PDBContainer.to_rdkit`.
    * hdf5-interconversion: :meth:`~PDBContainer.create_hdf5_group`,
      :meth:`~PDBContainer.validate_hdf5`,
      :meth:`~PDBContainer.to_hdf5` & :meth:`~PDBContainer.from_hdf5`.
    * Miscellaneous: :meth:`~PDBContainer.keys`, :meth:`~PDBContainer.values`,
      :meth:`~PDBContainer.items`, :meth:`~PDBContainer.__getitem__` &
      :meth:`~PDBContainer.__len__`.

    Examples
    --------
    .. testsetup:: python

        >>> import os

        >>> from dataCAT.testing_utils import (
        ...     MOL_TUPLE as mol_list,
        ...     PDB as pdb,
        ...     HDF5_TMP as hdf5_file
        ... )

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    .. code:: python

        >>> import h5py
        >>> from scm.plams import readpdb
        >>> from dataCAT import PDBContainer

        >>> mol_list [readpdb(...), ...]  # doctest: +SKIP
        >>> pdb = PDBContainer.from_molecules(mol_list)
        >>> print(pdb)
        PDBContainer(
            atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
            bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
            atom_count = numpy.ndarray(..., shape=(23,), dtype=int32),
            bond_count = numpy.ndarray(..., shape=(23,), dtype=int32),
            scale      = numpy.recarray(..., shape=(23,), dtype=...)
        )

        >>> hdf5_file = str(...)  # doctest: +SKIP
        >>> with h5py.File(hdf5_file, 'a') as f:
        ...     group = pdb.create_hdf5_group(f, name='ligand')
        ...     pdb.to_hdf5(group, None)
        ...
        ...     print('group', '=', group)
        ...     for name, dset in group.items():
        ...         print(f'group[{name!r}]', '=', dset)
        group = <HDF5 group "/ligand" (5 members)>
        group['atoms'] = <HDF5 dataset "atoms": shape (23, 76), type "|V46">
        group['bonds'] = <HDF5 dataset "bonds": shape (23, 75), type "|V9">
        group['atom_count'] = <HDF5 dataset "atom_count": shape (23,), type "<i4">
        group['bond_count'] = <HDF5 dataset "bond_count": shape (23,), type "<i4">
        group['index'] = <HDF5 dataset "index": shape (23,), type "<i4">

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    """

    __slots__ = (
        '__weakref__', '_hash', '_atoms', '_bonds', '_atom_count', '_bond_count', '_scale'
    )

    #: A mapping holding the dimensionality of each array embedded within this class.
    NDIM: ClassVar[Mapping[_AttrName, int]] = MappingProxyType({
        'atoms': 2,
        'bonds': 2,
        'atom_count': 1,
        'bond_count': 1,
        'scale': 1
    })

    #: A mapping holding the dtype of each array embedded within this class.
    DTYPE: ClassVar[Mapping[_AttrName, np.dtype]] = MappingProxyType({
        'atoms': ATOMS_DTYPE,
        'bonds': BONDS_DTYPE,
        'atom_count': ATOM_COUNT_DTYPE,
        'bond_count': BOND_COUNT_DTYPE,
        'scale': BACKUP_IDX_DTYPE
    })

    #: The name of the h5py dimensional scale.
    SCALE_NAME: ClassVar[str] = 'index'

    @property
    def atoms(self) -> np.recarray:
        """:class:`numpy.recarray`, shape :math:`(n, m)`: Get a read-only padded recarray for keeping track of all atom-related information.

        See :data:`dataCAT.dtype.ATOMS_DTYPE` for a comprehensive overview of
        all field names and dtypes.

        """  # noqa: E501
        return self._atoms

    @property
    def bonds(self) -> np.recarray:
        """:class:`numpy.recarray`, shape :math:`(n, k)` : Get a read-only padded recarray for keeping track of all bond-related information.

        Note that all atomic indices are 1-based.

        See :data:`dataCAT.dtype.BONDS_DTYPE` for a comprehensive overview of
        all field names and dtypes.

        """  # noqa: E501
        return self._bonds

    @property
    def atom_count(self) -> np.ndarray:
        """:class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)` : Get a read-only ndarray for keeping track of the number of atoms in each molecule in :attr:`~PDBContainer.atoms`."""  # noqa: E501
        return self._atom_count

    @property
    def bond_count(self) -> np.ndarray:
        """:class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)` : Get a read-only ndarray for keeping track of the number of atoms in each molecule in :attr:`~PDBContainer.bonds`."""  # noqa: E501
        return self._bond_count

    @property
    def scale(self) -> np.recarray:
        """:class:`numpy.recarray`, shape :math:`(n,)`: Get a recarray representing an index.

        Used as dimensional scale in the h5py Group.

        """  # noqa: E501
        return self._scale

    @overload
    def __init__(self, atoms: np.recarray, bonds: np.recarray,
                 atom_count: np.ndarray, bond_count: np.ndarray,
                 scale: np.recarray,
                 validate: Literal[False]) -> None:
        ...
    @overload  # noqa: E301
    def __init__(self, atoms: ArrayLike, bonds: ArrayLike,
                 atom_count: ArrayLike, bond_count: ArrayLike,
                 scale: ArrayLike | None = None,
                 validate: Literal[True] = ..., copy: bool = ...,) -> None:
        ...
    def __init__(self, atoms, bonds, atom_count, bond_count, scale=None, validate=True, copy=True, index_dtype=None):  # noqa: E501,E301
        """Initialize an instance.

        Parameters
        ----------
        atoms : :class:`numpy.recarray`, shape :math:`(n, m)`
            A padded recarray for keeping track of all atom-related information.
            See :attr:`PDBContainer.atoms`.
        bonds : :class:`numpy.recarray`, shape :math:`(n, k)`
            A padded recarray for keeping track of all bond-related information.
            See :attr:`PDBContainer.bonds`.
        atom_count : :class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)`
            An ndarray for keeping track of the number of atoms in each molecule in **atoms**.
            See :attr:`PDBContainer.atom_count`.
        bond_count : :class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)`
            An ndarray for keeping track of the number of bonds in each molecule in **bonds**.
            See :attr:`PDBContainer.bond_count`.
        scale : :class:`numpy.recarray`, shape :math:`(n,)`, optional
            A recarray representing an index.
            If :data:`None`, use a simple numerical index (*i.e.* :func:`numpy.arange`).
            See :attr:`PDBContainer.scale`.

        Keyword Arguments
        -----------------
        validate : :class:`bool`
            If :data:`True` perform more thorough validation of the input arrays.
            Note that this also allows the parameters to-be passed as array-like objects
            in addition to aforementioned :class:`~numpy.ndarray` or
            :class:`~numpy.recarray` instances.
        copy : :class:`bool`
            If :data:`True`, set the passed arrays as copies.
            Only relevant if :data:`validate = True<True>`.


        :rtype: :data:`None`

        """
        if validate:
            cls = type(self)
            rec_set = {'atoms', 'bonds'}
            items = [
                ('atoms', atoms),
                ('bonds', bonds),
                ('atom_count', atom_count),
                ('bond_count', bond_count)
            ]

            _copy = True if copy else _COPY_IF_NEEDED
            for name, _array in items:
                ndmin = cls.NDIM[name]
                dtype = cls.DTYPE[name]

                array = np.array(_array, dtype=dtype, ndmin=ndmin, copy=_copy)
                if name in rec_set:
                    array = array.view(np.recarray)
                setattr(self, f'_{name}', array)

            if scale is None:
                dtype = cls.DTYPE['scale']
                self._scale: np.recarray = np.arange(len(array), dtype=dtype).view(np.recarray)
            else:
                self._scale = np.array(scale, ndmin=1, copy=_copy).view(np.recarray)

            len_set = {len(ar) for ar in self.values()}
            if len(len_set) != 1:
                raise ValueError("All passed arrays should be of the same length")

        # Assume the input does not have to be parsed
        else:
            self._atoms: np.recarray = atoms
            self._bonds: np.recarray = bonds
            self._atom_count: np.ndarray = atom_count
            self._bond_count: np.ndarray = bond_count
            self._scale: np.recarray = scale

        for ar in self.values():
            ar.setflags(write=False)

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        wdith = max(len(k) for k in self.keys())

        def _str(k, v):
            dtype = '...' if isinstance(v, np.recarray) else str(v.dtype)

            # NumPy >= 2.0 reports `numpy.rec` as the module of `numpy.recarray`
            cls = type(v)
            module = 'numpy' if cls.__module__.startswith('numpy') else cls.__module__
            return f'{k:{wdith}} = {module}.{cls.__name__}(..., shape={v.shape}, dtype={dtype})'

        ret = ',\n'.join(_str(k, v) for k, v in self.items())
        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent)}\n)'

    def __reduce__(self: ST) -> tuple[type[ST], _ReduceTuple]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        return cls, (*self.values(), False)  # type: ignore

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self  # self is immutable

    def __deepcopy__(self: ST, memo: dict[int, Any] | None = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self  # self is immutable

    def __len__(self) -> int:
        """Implement :func:`len(self)<len>`.

        Returns
        -------
        :class:`int`
            Returns the length of the arrays embedded within this instance
            (which are all of the same length).

        """
        return len(self.atom_count)

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value) or hash(self) != hash(value):
            return False

        iterator = ((v, getattr(value, k)) for k, v in self.items())
        return all(np.all(ar1 == ar2) for ar1, ar2 in iterator)

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            args = []

            # The hash of each individual array consists of its shape appended
            # with the array's first and last element along axis 0
            for ar in self.values():
                if not len(ar):
                    first_and_last: tuple[Any, ...] = ()
                else:
                    first = ar[0] if ar.ndim == 1 else ar[0, 0]
                    last = ar[-1] if ar.ndim == 1 else ar[-1, 0]
                    first_and_last = (first, last)
                args.append(ar.shape + first_and_last)

            cls = type(self)
            self._hash: int = hash((cls, tuple(args)))
            return self._hash

    def __getitem__(self: ST, index: IndexLike) -> ST:
        """Implement :meth:`self[index]<object.__getitem__>`.

        Constructs a new :class:`PDBContainer` instance by slicing all arrays with **index**.
        Follows the standard NumPy broadcasting rules:
        if an integer or slice is passed then a shallow copy is returned;
        otherwise a deep copy will be created.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> print(pdb)
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(23,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(23,), dtype=int32),
                scale      = numpy.recarray(..., shape=(23,), dtype=...)
            )

            >>> pdb[0]
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(1, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(1, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(1,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(1,), dtype=int32),
                scale      = numpy.recarray(..., shape=(1,), dtype=...)
            )

            >>> pdb[:10]
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(10, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(10, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(10,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(10,), dtype=int32),
                scale      = numpy.recarray(..., shape=(10,), dtype=...)
            )

            >>> pdb[[0, 5, 7, 9, 10]]
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(5, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(5, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(5,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(5,), dtype=int32),
                scale      = numpy.recarray(..., shape=(5,), dtype=...)
            )

        Parameters
        ----------
        index : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`
            An object for slicing arrays along :code:`axis=0`.

        Returns
        -------
        :class:`dataCAT.PDBContainer`
            A shallow or deep copy of a slice of this instance.

        """
        cls = type(self)
        if index is None:
            idx: slice | np.ndarray = slice(None)
        else:
            try:
                idx = int_to_slice(index, len(self))  # type: ignore
            except (AttributeError, TypeError):
                idx = np.asarray(index) if not isinstance(index, slice) else index
                assert getattr(index, 'ndim', 1) == 1

        iterator = (ar[idx] for ar in self.values())
        ret: ST = cls(*iterator, validate=False)  # type: ignore
        ret._scale = ret.scale.view(np.recarray)
        return ret

    @classmethod
    def keys(cls) -> Generator[_AttrName, None, None]:
        """Yield the (public) attribute names in this class.

        Examples
        --------
        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> for name in PDBContainer.keys():
            ...     print(name)
            atoms
            bonds
            atom_count
            bond_count
            scale

        Yields
        ------
        :class:`str`
            The names of all attributes in this class.

        """
        return (name.strip('_') for name in cls.__slots__[2:])  # type: ignore

    def values(self) -> Generator[np.ndarray | np.recarray, None, None]:
        """Yield the (public) attributes in this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> for value in pdb.values():
            ...     print(type(value).__name__)
            recarray
            recarray
            ndarray
            ndarray
            recarray

        Yields
        ------
        :class:`str`
            The values of all attributes in this instance.

        """
        cls = type(self)
        return (getattr(self, name) for name in cls.keys())

    def items(self) -> Generator[tuple[_AttrName, np.ndarray | np.recarray], None, None]:
        """Yield the (public) attribute name/value pairs in this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> for name, value in pdb.items():
            ...     print(name, '=', type(value).__name__)
            atoms = recarray
            bonds = recarray
            atom_count = ndarray
            bond_count = ndarray
            scale = recarray

        Yields
        ------
        :class:`str` and :class:`numpy.ndarray` / :class:`numpy.recarray`
            The names and values of all attributes in this instance.

        """
        return ((n, getattr(self, n)) for n in self.keys())

    def concatenate(self: ST, *args: ST) -> ST:
        r"""Concatenate :math:`n` PDBContainers into a single new instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb1

            >>> pdb2 = pdb3 = pdb1

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb1 = PDBContainer(...)  # doctest: +SKIP
            >>> pdb2 = PDBContainer(...)  # doctest: +SKIP
            >>> pdb3 = PDBContainer(...)  # doctest: +SKIP
            >>> print(len(pdb1), len(pdb2), len(pdb3))
            23 23 23

            >>> pdb_new = pdb1.concatenate(pdb2, pdb3)
            >>> print(pdb_new)
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(69, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(69, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(69,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(69,), dtype=int32),
                scale      = numpy.recarray(..., shape=(69,), dtype=...)
            )

        Parameters
        ----------
        \*args : :class:`PDBContainer`
            One or more PDBContainers.

        Returns
        -------
        :class:`PDBContainer`
            A new PDBContainer cosntructed by concatenating **self** and **args**.

        """
        if not args:
            return self

        try:
            attr_list = [(k, v, [getattr(a, k) for a in args]) for k, v in self.items()]
        except AttributeError as ex:
            raise TypeError("'*args' expected one or more PDBContainer instances") from ex

        cls = type(self)
        ret_list = []
        for k, ar_self, ar_list in attr_list:
            # 'scale': a 1D recarray
            if k == 'scale':
                dtype = ar_self.dtype
                ar_list = [ar.astype(dtype, copy=False) for ar in ar_list]
                ar_new: np.ndarray = np.concatenate((ar_self, *ar_list)).view(np.recarray)

            # 'atom_count' and 'bond_count': two normal 1d arrays
            elif ar_self.ndim == 1:
                ar_new = np.concatenate((ar_self, *ar_list))

            # 'atoms' and 'bonds': two padded 2D recarrays
            else:
                ax0 = len(ar_self) + sum(len(ar) for ar in ar_list)
                ax1 = max(ar_self.shape[1], *(ar.shape[1] for ar in ar_list))
                ar_new = np.zeros((ax0, ax1), dtype=cls.DTYPE[k]).view(np.recarray)

                i = 0
                for ar in chain([ar_self], ar_list):
                    j = i + len(ar)
                    ar_new[i:j, :ar.shape[1]] = ar
                    i = j

            ret_list.append(ar_new)
        return cls(*ret_list, validate=False)  # type: ignore

    @classmethod
    def from_molecules(cls: type[ST], mol_list: Iterable[Molecule],
                       min_atom: int = 0,
                       min_bond: int = 0,
                       scale: ArrayLike | None = None) -> ST:
        """Convert an iterable or sequence of molecules into a new :class:`PDBContainer` instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import (
            ...     PDB as pdb,
            ...     MOL_TUPLE as mol_list
            ... )

        .. code:: python

            >>> from typing import List
            >>> from dataCAT import PDBContainer
            >>> from scm.plams import readpdb, Molecule

            >>> mol_list: List[Molecule] = [readpdb(...), ...]  # doctest: +SKIP
            >>> PDBContainer.from_molecules(mol_list)
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(23,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(23,), dtype=int32),
                scale      = numpy.recarray(..., shape=(23,), dtype=...)
            )

        Parameters
        ----------
        mol_list : :class:`Iterable[Molecule]<typing.Iterable>`
            An iterable consisting of PLAMS molecules.
        min_atom : :class:`int`
            The minimum number of atoms which :attr:`PDBContainer.atoms` should accomodate.
        min_bond : :class:`int`
            The minimum number of bonds which :attr:`PDBContainer.bonds` should accomodate.
        scale : array-like, optional
            An array-like object representing an user-specified index.
            Defaults to a simple range index if :data:`None` (see :func:`numpy.arange`).

        Returns
        -------
        :class:`dataCAT.PDBContainer`
            A pdb container.

        """
        try:
            mol_count = len(mol_list)  # type: ignore
        except TypeError:
            mol_list = list(mol_list)
            mol_count = len(mol_list)

        # Parse the index
        if scale is None:
            idx = np.arange(mol_count, dtype=cls.DTYPE['scale']).view(np.recarray)
        else:
            idx = np.asarray(scale).view(np.recarray)
        if len(idx) != mol_count:
            raise ValueError("'mol_list' and 'idx' must be of equal length")

        # Gather the shape of the to-be created atom (pdb-file) array
        _atom_count = max((len(mol.atoms) for mol in mol_list), default=0)
        atom_count = max(_atom_count, min_atom)
        atom_shape = mol_count, atom_count

        # Gather the shape of the to-be created bond array
        _bond_count = max((len(mol.bonds) for mol in mol_list), default=0)
        bond_count = max(_bond_count, min_bond)
        bond_shape = mol_count, bond_count

        # Construct the to-be returned (padded) arrays
        DTYPE = cls.DTYPE
        atom_array = np.zeros(atom_shape, dtype=DTYPE['atoms']).view(np.recarray)
        bond_array = np.zeros(bond_shape, dtype=DTYPE['bonds']).view(np.recarray)
        atom_counter = np.empty(mol_count, dtype=DTYPE['atom_count'])
        bond_counter = np.empty(mol_count, dtype=DTYPE['bond_count'])

        # Fill the to-be returned arrays
        for i, mol in enumerate(mol_list):
            j_atom = len(mol.atoms)
            j_bond = len(mol.bonds)

            atom_array[i, :j_atom] = [_get_atom_info(at, k) for k, at in enumerate(mol, 1)]
            bond_array[i, :j_bond] = _get_bond_info(mol)
            atom_counter[i] = j_atom
            bond_counter[i] = j_bond

        return cls(
            atoms=atom_array, bonds=bond_array,
            atom_count=atom_counter, bond_count=bond_counter,
            scale=idx, validate=False
        )

    @overload
    def to_molecules(self, index: None | Sequence[int] | slice | np.ndarray = ...,
                     mol: Iterable[Molecule | None] | None = ...) -> list[Molecule]:
        ...
    @overload  # noqa: E301
    def to_molecules(self, index: SupportsIndex = ..., mol: Molecule | None = ...) -> Molecule:
        ...
    def to_molecules(self, index=None, mol=None):  # noqa: E301
        """Create a molecule or list of molecules from this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import (
            ...     PDB as pdb,
            ...     MOL_TUPLE as mol_list,
            ...     MOL as mol
            ... )

        An example where one or more new molecules are created.

        .. code:: python

            >>> from dataCAT import PDBContainer
            >>> from scm.plams import Molecule

            >>> pdb = PDBContainer(...)  # doctest: +SKIP

            # Create a single new molecule from `pdb`
            >>> pdb.to_molecules(index=0)  # doctest: +ELLIPSIS
            <scm.plams.mol.molecule.Molecule object at ...>

            # Create three new molecules from `pdb`
            >>> pdb.to_molecules(index=[0, 1])  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
            [<scm.plams.mol.molecule.Molecule object at ...>,
             <scm.plams.mol.molecule.Molecule object at ...>]

        An example where one or more existing molecules are updated in-place.

        .. code:: python

            # Update `mol` with the info from `pdb`
            >>> mol = Molecule(...)  # doctest: +SKIP
            >>> mol_new = pdb.to_molecules(index=2, mol=mol)
            >>> mol is mol_new
            True

            # Update all molecules in `mol_list` with info from `pdb`
            >>> mol_list = [Molecule(...), Molecule(...), Molecule(...)]  # doctest: +SKIP
            >>> mol_list_new = pdb.to_molecules(index=range(3), mol=mol_list)
            >>> for m, m_new in zip(mol_list, mol_list_new):
            ...     print(m is m_new)
            True
            True
            True

        Parameters
        ----------
        index : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`, optional
            An object for slicing the arrays embedded within this instance.
            Follows the standard numpy broadcasting rules (*e.g.* :code:`self.atoms[index]`).
            If a scalar is provided (*e.g.* an integer) then a single molecule will be returned.
            If a sequence, range, slice, *etc.* is provided then
            a list of molecules will be returned.
        mol : :class:`~scm.plams.mol.molecule.Molecule` or :class:`Iterable[Molecule]<typing.Iterable>`, optional
            A molecule or list of molecules.
            If one or molecules are provided here then they will be updated in-place.

        Returns
        -------
        :class:`~scm.plams.mol.molecule.Molecule` or :class:`List[Molecule]<typing.List>`
            A molecule or list of molecules,
            depending on whether or not **index** is a scalar or sequence / slice.
            Note that if :data:`mol is not None<None>`, then the-be returned molecules won't be copies.

        """  # noqa: E501
        if index is None:
            i = slice(None)
            is_seq = True
        else:
            try:
                i = index.__index__()
                is_seq = False
            except (AttributeError, TypeError):
                i = index
                is_seq = True

        atoms = self.atoms[i]
        bonds = self.bonds[i]
        atom_count = self.atom_count[i]
        bond_count = self.bond_count[i]

        if not is_seq:
            return _rec_to_plams(atoms, bonds, atom_count, bond_count, mol)

        if mol is None:
            mol_list = repeat(None)
        elif isinstance(mol, Molecule):
            raise TypeError
        else:
            mol_list = mol

        iterator = zip(atoms, bonds, atom_count, bond_count, mol_list)
        return [_rec_to_plams(*args) for args in iterator]

    @overload
    def to_rdkit(  # type: ignore[overload-overlap]
        self,
        index: None | Sequence[int] | slice | np.ndarray = ...,
        sanitize: bool = ...,
    ) -> list[Chem.Mol]:
        ...
    @overload  # noqa: E301
    def to_rdkit(
        self,
        index: SupportsIndex,
        sanitize: bool = ...,
    ) -> Chem.Mol:
        ...
    def to_rdkit(self, index=None, sanitize=True):  # noqa: E301
        """Create an rdkit molecule or list of rdkit molecules from this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        An example where one or more new molecules are created.

        .. code:: python

            >>> from dataCAT import PDBContainer
            >>> from rdkit.Chem import Mol

            >>> pdb = PDBContainer(...)  # doctest: +SKIP

            # Create a single new molecule from `pdb`
            >>> pdb.to_rdkit(index=0)  # doctest: +ELLIPSIS
            <rdkit.Chem.rdchem.Mol object at ...>

            # Create three new molecules from `pdb`
            >>> pdb.to_rdkit(index=[0, 1])  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
            [<rdkit.Chem.rdchem.Mol object at ...>,
             <rdkit.Chem.rdchem.Mol object at ...>]

        Parameters
        ----------
        index : :class:`int`, :class:`Sequence[int]<Collections.abc.Sequence>` or :class:`slice`, optional
            An object for slicing the arrays embedded within this instance.
            Follows the standard numpy broadcasting rules (*e.g.* :code:`self.atoms[index]`).
            If a scalar is provided (*e.g.* an integer) then a single molecule will be returned.
            If a sequence, range, slice, *etc.* is provided then
            a list of molecules will be returned.
        sanitize : bool
            Whether to sanitize the molecule before returning or not.

        Returns
        -------
        :class:`~rdkit.Chem.rdchem.Mol` or :class:`list[Mol]<list>`
            A molecule or list of molecules,
            depending on whether or not **index** is a scalar or sequence / slice.

        """  # noqa: E501
        if index is None:
            i = slice(None)
            is_seq = True
        else:
            try:
                i = index.__index__()
                is_seq = False
            except (AttributeError, TypeError):
                i = index
                is_seq = True

        atoms = self.atoms[i]
        bonds = self.bonds[i]
        atom_count = self.atom_count[i]
        bond_count = self.bond_count[i]

        if not is_seq:
            return _rec_to_rdkit(atoms, bonds, atom_count, bond_count, sanitize)

        iterator = zip(atoms, bonds, atom_count, bond_count, repeat(sanitize))
        return [_rec_to_rdkit(*args) for args in iterator]

    @overload
    @classmethod
    def create_hdf5_group(cls, file: h5py.Group, name: str, *,
                          scale: h5py.Dataset, **kwargs: Any) -> h5py.Group:
        ...
    @overload  # noqa: E301
    @classmethod
    def create_hdf5_group(cls, file: h5py.Group, name: str, *,
                          scale_dtype: DTypeLike = ..., **kwargs: Any) -> h5py.Group:
        ...
    @classmethod  # noqa: E301
    def create_hdf5_group(cls, file, name, *, scale=None, scale_dtype=None, **kwargs):
        r"""Create a h5py Group for storing :class:`dataCAT.PDBContainer` instances.

        Notes
        -----
        The **scale** and **scale_dtype** parameters are mutually exclusive.

        Parameters
        ----------
        file : :class:`h5py.File` or :class:`h5py.Group`
            The h5py File or Group where the new Group will be created.
        name : :class:`str`
            The name of the to-be created Group.

        Keyword Arguments
        -----------------
        scale : :class:`h5py.Dataset`, keyword-only
            A pre-existing dataset serving as dimensional scale.
            See **scale_dtype** to create a new instead instead.
        scale_dtype : dtype-like, keyword-only
            The datatype of the to-be created dimensional scale.
            See **scale** to use a pre-existing dataset for this purpose.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for the creation of each dataset.
            Arguments already specified by default are:
            ``name``, ``shape``, ``maxshape`` and ``dtype``.

        Returns
        -------
        :class:`h5py.Group`
            The newly created Group.

        """
        if scale is not None and scale_dtype is not None:
            raise TypeError("'scale' and 'scale_dtype' cannot be both specified")

        cls_name = cls.__name__

        # Create the group
        grp = file.create_group(name, track_order=True)
        grp.attrs['__doc__'] = np.bytes_(HDF5_DOCSTRING.format(cls_name=cls_name))

        # Create the datasets
        NDIM = cls.NDIM
        DTYPE = cls.DTYPE
        key_iter = (k for k in cls.keys() if k != 'scale')
        for key in key_iter:
            maxshape = NDIM[key] * (None,)
            shape = NDIM[key] * (0,)
            dtype = DTYPE[key]

            dset = grp.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)
            dset.attrs['__doc__'] = np.bytes_(f"A dataset representing `{cls_name}.atoms`.")

        # Set the index
        scale_name = cls.SCALE_NAME
        if scale is not None:
            scale_dset = scale
        else:
            _dtype = scale_dtype if scale_dtype is not None else cls.DTYPE['scale']
            scale_dset = grp.create_dataset(scale_name, shape=(0,), maxshape=(None,), dtype=_dtype)
            scale_dset.make_scale(scale_name)

        # Use the index as a scale
        dset_iter = (grp[k] for k in cls.keys() if k != 'scale')
        for dset in dset_iter:
            dset.dims[0].label = scale_name
            dset.dims[0].attach_scale(scale_dset)

        grp['atoms'].dims[1].label = 'atoms'
        grp['bonds'].dims[1].label = 'bonds'
        return grp

    @classmethod
    def validate_hdf5(cls, group: h5py.Group) -> None:
        """Validate the passed hdf5 **group**, ensuring it is compatible with :class:`PDBContainer` instances.

        An :exc:`AssertionError` will be raise if **group** does not validate.

        This method is called automatically when an exception is raised by
        :meth:`~PDBContainer.to_hdf5` or :meth:`~PDBContainer.from_hdf5`.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be validated hdf5 Group.

        Raises
        ------
        :exc:`AssertionError`
            Raised if the validation process fails.

        """  # noqa: E501
        if not isinstance(group, h5py.Group):
            raise TypeError("'group' expected a h5py.Group; "
                            f"observed type: {group.__class__.__name__}")

        # Check if **group** has all required keys
        keys: list[str] = list(cls.keys())
        keys[-1] = cls.SCALE_NAME
        difference = set(keys) - group.keys()
        if difference:
            missing_keys = ', '.join(repr(i) for i in difference)
            raise AssertionError(f"Missing keys in {group!r}: {missing_keys}")

        # Check the dimensionality and dtype of all datasets
        len_dict = {}
        for key in cls.keys():
            dset = group[key] if key != 'scale' else group[cls.SCALE_NAME]

            len_dict[key] = len(dset)
            assertion.eq(dset.ndim, cls.NDIM[key], message=f"{key!r} ndim mismatch")
            if key != 'scale':
                assertion.eq(dset.dtype, cls.DTYPE[key], message=f"{key!r} dtype mismatch")

        # Check that all datasets are of the same length
        if len(set(len_dict.values())) != 1:
            raise AssertionError(
                f"All datasets in {group!r} should be of the same length.\n"
                f"Observed lengths: {len_dict!r}"
            )

    @if_exception(validate_hdf5.__func__)  # type: ignore
    def to_hdf5(self, group: h5py.Group, index: IndexLike, update_scale: bool = True) -> None:
        """Update the datasets in **group** at **index** with their counterparts in this instance.

        Follows the standard broadcasting rules as employed by h5py.

        Important
        ---------
        If **index** is passed as a sequence of integers then, contrary to NumPy,
        they *will* have to be sorted.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be updated h5py group.
        index : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`
            An object for slicing all datasets in **group**.
            Note that, contrary to numpy, if a sequence of integers is provided
            then they'll have to ordered.
        update_scale : :class:`bool`
            If :data:`True`, also export :attr:`PDBContainer.scale` to the dimensional
            scale in the passed **group**.

        """
        # Parse **idx**
        if index is None:
            idx: slice | np.ndarray = slice(None)
            idx_max: int | np.integer = len(self)
        else:
            try:
                idx = int_to_slice(index, len(self))  # type: ignore
                idx_max = idx.stop or len(self)
            except (AttributeError, TypeError):
                if not isinstance(index, slice):
                    idx = np.asarray(index)
                    idx_max = idx.max()
                    assert idx.ndim == 1
                    assert issubclass(idx.dtype.type, np.integer)
                else:
                    idx = index
                    idx_max = idx.stop or len(self)

        # Update the length of all groups
        scale_name = self.SCALE_NAME
        scale = group['atoms'].dims[0][scale_name]
        if len(scale) < idx_max:
            scale.resize(idx_max, axis=0)
        self._update_hdf5_shape(group)

        # Update the datasets
        items = ((name, ar) for name, ar in self.items() if name != 'scale')
        for name, ar in items:
            dataset = group[name]

            if ar.ndim == 1:
                dataset[idx] = ar
            else:
                _, j = ar.shape
                dataset[idx, :j] = ar

        # Update the index
        if update_scale:
            scale[idx] = self.scale

    def _update_hdf5_shape(self, group: h5py.Group) -> None:
        """Update the shape of all datasets in **group** such that it can accommodate **pdb**.

        The length of all datasets will be set equal to the ``index`` dimensional cale.

        This method is automatically called by :meth:`PDBContainer.update_hdf5` when required.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The h5py Group with the to-be reshaped datasets.
        pdb : :class:`dataCAT.PDBContainer`
            The pdb container for updating **group**.


        :rtype: :data:`None`

        """  # noqa: E501
        # Get the length of the dimensional scale
        scale_name = self.SCALE_NAME
        idx = group['atoms'].dims[0][scale_name]
        idx_len = len(idx)

        items = ((name, ar) for name, ar in self.items() if name != 'scale')
        for name, ar in items:
            dataset = group[name]

            # Identify the new shape of all datasets
            shape = np.fromiter(dataset.shape, dtype=int)
            shape[0] = idx_len
            if ar.ndim == 2:
                shape[1] = max(shape[1], ar.shape[1])

            # Set the new shape
            dataset.shape = shape

    @classmethod
    @if_exception(validate_hdf5.__func__)  # type: ignore
    def from_hdf5(cls: type[ST], group: h5py.Group, index: IndexLike = None) -> ST:
        """Construct a new PDBContainer from the passed hdf5 **group**.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be read h5py group.
        index : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`, optional
            An object for slicing all datasets in **group**.

        Returns
        -------
        :class:`dataCAT.PDBContainer`
            A new PDBContainer constructed from **group**.

        """
        if index is None:
            idx: slice | np.ndarray = slice(None)
        else:
            try:
                idx = int_to_slice(index, len(group['atom_count']))  # type: ignore
            except (AttributeError, TypeError):
                idx = np.asarray(index) if not isinstance(index, slice) else index
                assert getattr(idx, 'ndim', 1) == 1

        scale_name = cls.SCALE_NAME
        return cls(
            atoms=group['atoms'][idx].view(np.recarray),
            bonds=group['bonds'][idx].view(np.recarray),
            atom_count=group['atom_count'][idx],
            bond_count=group['bond_count'][idx],
            scale=group[scale_name][idx].view(np.recarray),
            validate=False
        )

    def intersection(self: ST, value: ST | ArrayLike) -> ST:
        """Construct a new PDBContainer by the intersection of **self** and **value**.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        An example where one or more new molecules are created.

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> print(pdb.scale)
            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]

            >>> pdb_new = pdb.intersection(range(4))
            >>> print(pdb_new.scale)
            [0 1 2 3]

        Parameters
        ----------
        value : :class:`PDBContainer` or array-like
            Another PDBContainer or an array-like object representing :attr:`PDBContainer.scale`.
            Note that both **value** and **self.scale** should consist of unique elements.

        Returns
        -------
        :class:`PDBContainer`
            A new instance by intersecting :attr:`self.scale<PDBContainer.scale>` and **value**.

        See Also
        --------
        :meth:`set.intersection<frozenset.intersection>`
            Return the intersection of two sets as a new set.

        """
        idx1, idx2 = self._get_index(value)
        _, i, _ = np.intersect1d(idx1, idx2, assume_unique=True, return_indices=True)
        return self[i]

    def difference(self: ST, value: ST | ArrayLike) -> ST:
        """Construct a new PDBContainer by the difference of **self** and **value**.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> print(pdb.scale)
            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]

            >>> pdb_new = pdb.difference(range(10, 30))
            >>> print(pdb_new.scale)
            [0 1 2 3 4 5 6 7 8 9]

        Parameters
        ----------
        value : :class:`PDBContainer` or array-like
            Another PDBContainer or an array-like object representing :attr:`PDBContainer.scale`.
            Note that both **value** and **self.scale** should consist of unique elements.

        Returns
        -------
        :class:`PDBContainer`
            A new instance as the difference of :attr:`self.scale<PDBContainer.scale>`
            and **value**.

        See Also
        --------
        :meth:`set.difference<frozenset.difference>`
            Return the difference of two or more sets as a new set.

        """
        idx1, idx2 = self._get_index(value)
        i = np.isin(idx1, idx2, assume_unique=True, invert=True)
        return self[i]

    def symmetric_difference(self: ST, value: ST) -> ST:
        """Construct a new PDBContainer by the symmetric difference of **self** and **value**.

        Examples
        --------
        .. testsetup:: python

            >>> import numpy as np
            >>> from dataCAT.testing_utils import PDB as pdb
            >>> from dataCAT import PDBContainer

            >>> a = np.arange(10, 30)
            >>> b = a.reshape(len(a), 1)
            >>> pdb2 = PDBContainer(b, b, a, a, a, validate=False)

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> pdb2 = PDBContainer(..., scale=range(10, 30))  # doctest: +SKIP

            >>> print(pdb.scale)
            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]

            >>> pdb_new = pdb.symmetric_difference(pdb2)
            >>> print(pdb_new.scale)
            [ 0  1  2  3  4  5  6  7  8  9 23 24 25 26 27 28 29]

        Parameters
        ----------
        value : :class:`PDBContainer`
            Another PDBContainer.
            Note that both **value.scale** and **self.scale** should consist of unique elements.

        Returns
        -------
        :class:`PDBContainer`
            A new instance as the symmetric difference of :attr:`self.scale<PDBContainer.scale>`
            and **value**.

        See Also
        --------
        :meth:`set.symmetric_difference<frozenset.symmetric_difference>`
            Return the symmetric difference of two sets as a new set.

        """
        pdb1 = self.difference(value)
        pdb2 = value.difference(self)
        return pdb1.concatenate(pdb2)

    def union(self: ST, value: ST) -> ST:
        """Construct a new PDBContainer by the union of **self** and **value**.

        Examples
        --------
        .. testsetup:: python

            >>> import numpy as np
            >>> from dataCAT.testing_utils import PDB as pdb
            >>> from dataCAT import PDBContainer

            >>> a = np.arange(10, 30)
            >>> b = a.reshape(len(a), 1)
            >>> pdb2 = PDBContainer(b, b, a, a, a, validate=False)

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> pdb2 = PDBContainer(..., scale=range(10, 30))  # doctest: +SKIP

            >>> print(pdb.scale)
            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]

            >>> pdb_new = pdb.union(pdb2)
            >>> print(pdb_new.scale)
            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
             24 25 26 27 28 29]

        Parameters
        ----------
        value : :class:`PDBContainer`
            Another PDBContainer.
            Note that both **value** and **self.scale** should consist of unique elements.

        Returns
        -------
        :class:`PDBContainer`
            A new instance as the union of :attr:`self.index<PDBContainer.index>` and **value**.

        See Also
        --------
        :meth:`set.union<frozenset.union>`
            Return the union of sets as a new set.

        """
        cls = type(self)
        if not isinstance(value, cls):
            raise TypeError(f"'value' expected a {cls.__name__} instance; "
                            f"observed type: {value.__class__.__name__}")

        idx1 = self.scale
        idx2 = value.scale.astype(idx1.dtype, copy=False)
        i = np.isin(idx2, idx1, assume_unique=True, invert=True)

        ret = value[i]
        return self.concatenate(ret)

    def _get_index(self: ST, value: ST | ArrayLike) -> tuple[np.recarray, np.ndarray]:
        """Parse and return the :attr:`~PDBContainer.scale` of **self** and **value**."""
        cls = type(self)
        scale1 = self.scale
        _scale2 = value.scale if isinstance(value, cls) else value
        scale2 = np.asarray(_scale2, dtype=scale1.dtype)
        return scale1, scale2
