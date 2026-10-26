"""A module which holds the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    Database

API
---
.. autoclass:: Database
    :members:

"""

import reprlib
import textwrap
import warnings
import functools
from os import getcwd, PathLike
from os.path import abspath
from types import MappingProxyType
from functools import partial
from typing import (
    Union, Any, TypeVar,
    overload, TypedDict, cast, TYPE_CHECKING
)
from contextlib import AbstractContextManager
from collections.abc import Sequence, Mapping, Iterable, Callable

import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient, InsertOne, ReplaceOne
from pymongo.errors import PyMongoError, BulkWriteError

from rdkit.Chem import Mol
from scm.plams import Settings, Molecule, to_rdmol
from CAT.workflows import HDF5_INDEX, OPT, MOL

from .create_database import (
    create_csv, create_hdf5, create_mongodb, QD, Ligand, IDX_DTYPE, MONGO_COLLECTIONS
)
from .context_managers import OpenLig, OpenQD
from .locking import get_lock, open_hdf5
from .functions import df_to_mongo_dict, mongo_dict_to_df, even_index, hdf5_availability
from .pdb_array import PDBContainer
from .hdf5_log import update_hdf5_log
from .property_dset import create_prop_dset, update_prop_dset
from ._parse_settings import _update_hdf5_settings

if TYPE_CHECKING:
    from .df_proxy import DFProxy
    from numpy.typing import DTypeLike
else:
    DFProxy = 'dataCAT.DFProxy'
    DTypeLike = 'numpy.typing.DTypeLike'

__all__ = ['Database']

KT = TypeVar('KT')
ST = TypeVar('ST', bound='Database')
MIT = TypeVar('MIT', bound=pd.MultiIndex)


class JobRecipe(TypedDict):
    """A :class:`~typing.TypedDict` representing the input of :class:`.Database.update_yaml`."""

    key: str | type
    value: str | Settings


#: The MongoDB error code of a duplicate key error.
_DUPLICATE_KEY_CODE = 11000

FT = TypeVar('FT', bound=Callable[..., Any])


def _locked(func: FT) -> FT:
    """Decorate a :class:`Database` method, so that it holds the lock of the database throughout."""
    @functools.wraps(func)
    def wrapper(self: 'Database', *args: Any, **kwargs: Any) -> Any:
        with get_lock(self.dirname):
            return func(self, *args, **kwargs)
    return wrapper  # type: ignore[return-value]


def _parse_mongo_query(query: Mapping[Any, Any], add_fields: dict[str, Any]) -> dict[str, Any]:
    """Translate the ``(index, sub index)`` keys of **query** into MongoDB fields.

    Plain sub indices become dotted paths (``"index.sub"``). As dots cannot be escaped,
    sub indices that are empty or contain a dot are copied to temporary fields, which are
    added to **add_fields**; querying these keeps the usual MongoDB semantics for
    missing values and NaN (unlike comparisons within ``"$expr"``).

    """
    ret = {}
    for key, value in query.items():
        if key in ('$and', '$or', '$nor'):
            ret[key] = [_parse_mongo_query(i, add_fields) for i in value]
        elif not isinstance(key, tuple):
            ret[key] = value
        elif key[1] and '.' not in key[1] and not key[1].startswith('$'):
            ret[f'{key[0]}.{key[1]}'] = value
        else:
            name = f'_query{len(add_fields)}'
            add_fields[name] = {
                '$getField': {'field': {'$literal': key[1]}, 'input': f'${key[0]}'}
            }
            ret[name] = value
    return ret


def _to_int64_array(ar: np.ndarray) -> np.ndarray:
    ret = ar.astype(np.int64)
    if ar.dtype.kind in "SU":
        return ret

    is_eq = (ret == ar)
    if not isinstance(is_eq, (np.ndarray, np.bool_)) or not is_eq.all():
        raise TypeError(f"Cannot safelly cast {ar.dtype} to {ret.dtype}")
    return ret


class Database:
    """The Database class."""

    __slots__ = ('__weakref__', '_dirname', '_csv_lig', '_csv_qd', '_hdf5', '_mongodb', '_hash')

    @property
    def dirname(self) -> str:
        """Get the path+filename of the directory containing all database components."""
        return self._dirname

    @property
    def csv_lig(self) -> 'partial[OpenLig]':
        """:data:`Callable[..., dataCAT.OpenLig]<typing.Callable>`: Get a function for constructing an :class:`dataCAT.OpenLig` context manager."""  # noqa: E501
        return self._csv_lig

    @property
    def csv_qd(self) -> 'partial[OpenQD]':
        """:data:`Callable[..., dataCAT.OpenQD]<typing.Callable>`: Get a function for constructing an :class:`dataCAT.OpenQD` context manager."""  # noqa: E501
        return self._csv_qd

    @property
    def hdf5(self) -> 'partial[AbstractContextManager[h5py.File]]':
        """:data:`Callable[..., AbstractContextManager[h5py.File]]<typing.Callable>`: Get a function for opening the .hdf5 file as :class:`h5py.File`, while holding the lock of the database (see :mod:`dataCAT.locking`)."""  # noqa: E501
        return self._hdf5

    @property
    def mongodb(self) -> Mapping[str, Any] | None:
        """:class:`Mapping[str, Any]<typing.Mapping>`, optional: Get a mapping with keyword arguments for :class:`pymongo.MongoClient<pymongo.mongo_client.MongoClient>`."""  # noqa: E501
        return self._mongodb

    def __init__(self, path: Union[str, 'PathLike[str]', None] = None,
                 host: str = 'localhost',
                 port: int = 27017,
                 **kwargs) -> None:
        """Initialize :class:`Database`.

        Parameters
        ----------
        path : str
            The path+directory name of the directory which is to contain all database components
            (see :attr:`Database.dirname`).
        host : str
            Hostname or IP address or Unix domain socket path of a single mongod or
            mongos instance to connect to, or a mongodb URI, or a list of hostnames mongodb URIs.
            If **host** is an IPv6 literal it must be enclosed in ``"["`` and ``"]"`` characters
            following the RFC2732 URL syntax (e.g. ``"[::1]"`` for localhost).
            Multihomed and round robin DNS addresses are not supported.
            See :attr:`Database.mongodb`.
        port : str
            port number on which to connect.
            See :attr:`Database.mongodb`.
        **kwargs
            Optional keyword argument for :class:`pymongo.MongoClient<pymongo.mongo_client.MongoClient>`.
            See :attr:`Database.mongodb`.

        """  # noqa: E501
        self._dirname: str = abspath(path) if path is not None else getcwd()

        # Create the database components and return the filename
        with get_lock(self.dirname):
            lig_path = create_csv(self.dirname, database='ligand')
            qd_path = create_csv(self.dirname, database='qd')
            hdf5_path = create_hdf5(self.dirname)

        # Populate attributes with MetaManager instances
        self._csv_lig = partial(OpenLig, filename=lig_path)
        self._csv_qd = partial(OpenQD, filename=qd_path)
        self._hdf5 = cast(
            'partial[AbstractContextManager[h5py.File]]',
            partial(open_hdf5, hdf5_path, libver='latest'),
        )

        # Try to create or access the mongodb database
        try:
            self._mongodb: Mapping[str, Any] | None = MappingProxyType(
                create_mongodb(host, port, **kwargs)
            )
        except PyMongoError:
            self._mongodb = None

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        attr_tup = ('dirname', 'csv_lig', 'csv_qd', 'hdf5', 'mongodb')
        attr_max = max(len(i) for i in attr_tup)

        iterator = ((name, getattr(self, name)) for name in attr_tup[:-1])
        args = ',\n'.join(f'{name:{attr_max}} = {attr!r}' for name, attr in iterator)
        args += f',\n{attr_tup[-1]:{attr_max}} = {reprlib.repr(self.mongodb)}'

        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(args, indent)}\n)'

    def __eq__(self, value: Any) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False

        ret: bool = self.dirname == value.dirname and self.mongodb == value.mongodb
        if not ret:
            return False

        partial_names = ('csv_lig', 'csv_qd', 'hdf5')
        iterator = ((getattr(self, name), getattr(value, name)) for name in partial_names)
        return all(func1.args == func2.args and func1.keywords == func2.keywords and
                   func1.func is func2.func for func1, func2 in iterator)

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            cls, args, state = self.__reduce__()
            if state is not None:
                state = frozenset(state.items())  # type: ignore
            self._hash: int = hash((cls, args, state))
            return self._hash

    def __reduce__(self: ST) -> tuple[type[ST], tuple[str], dict[str, Any] | None]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        mongodb = self.mongodb if self.mongodb is None else dict(self.mongodb)
        return cls, (self.dirname,), mongodb

    def __setstate__(self, state: dict[str, Any] | None) -> None:
        """Helper for :mod:`pickle` and :meth:`~Database.__reduce__`."""
        if state is None:
            self._mongodb = None
            return

        try:
            self._mongodb = MappingProxyType(create_mongodb(**state))
        except PyMongoError:
            self._mongodb = None

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self

    def __deepcopy__(self: ST, memo: dict[int, Any] | None = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self

    """ #################################  Updating the database ############################## """

    @overload
    def _parse_database(self, database: Ligand) -> 'partial[OpenLig]':
        ...
    @overload  # noqa: E301
    def _parse_database(self, database: QD) -> 'partial[OpenQD]':
        ...
    def _parse_database(self, database):  # noqa: E301
        """Operate on either the ligand or quantum dot database."""
        if database in {'ligand', 'ligand_no_opt'}:
            return self.csv_lig
        elif database in {'qd', 'qd_no_opt'}:
            return self.csv_qd
        raise ValueError(f"database={database!r}; accepted values for are 'ligand' and 'qd'")

    @_locked
    def update_mongodb(self, database: str | Mapping[str, pd.DataFrame] = 'ligand',
                       overwrite: bool = False) -> None:
        """Export ligand or qd results to the MongoDB database.

        Examples
        --------
        .. code:: python

            >>> from dataCAT import Database

            >>> kwargs = dict(...)  # doctest: +SKIP
            >>> db = Database(**kwargs)  # doctest: +SKIP

            # Update from db.csv_lig
            >>> db.update_mongodb('ligand')  # doctest: +SKIP

            # Update from a lig_df, a user-provided DataFrame
            >>> db.update_mongodb({'ligand': lig_df})  # doctest: +SKIP
            >>> print(type(lig_df))  # doctest: +SKIP
            <class 'pandas.core.frame.DataFrame'>

        Parameters
        ----------
        database : :class:`str` or :class:`Mapping[str, pandas.DataFrame]<typing.Mapping>`
            The type of database.
            Accepted values are ``"ligand"`` and ``"qd"``,
            opening :attr:`Database.csv_lig` and :attr:`Database.csv_qd`, respectivelly.
            Alternativelly, a dictionary with the database name and a matching DataFrame
            can be passed directly.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.


        :rtype: :data:`None`

        """
        if self.mongodb is None:
            raise ValueError('Database.Mongodb is None')

        if callable(getattr(database, 'items', None)):
            name, df = next(iter(database.items()))  # type: ignore
        else:
            name = database
            df = None

        try:
            collection_name, idx_keys = MONGO_COLLECTIONS[name]
        except (KeyError, TypeError):
            raise ValueError(
                f"database={name!r}; accepted values for are 'ligand' and 'qd'"
            ) from None

        # Parse the ligand or qd dataframe
        if df is None:
            manager = self.csv_lig if name == 'ligand' else self.csv_qd
            with manager(write=False) as db:
                docs = df_to_mongo_dict(db.ndframe, as_gen=False)
        else:
            idx_keys = tuple(df.index.names)
            docs = df_to_mongo_dict(df, as_gen=False)
        if not docs:
            return None

        # Insert all new documents; replace pre-existing ones if `overwrite=True`
        requests: list[InsertOne | ReplaceOne]
        if overwrite:
            requests = [
                ReplaceOne({k: doc[k] for k in idx_keys}, doc, upsert=True) for doc in docs
            ]
        else:
            requests = [InsertOne(doc) for doc in docs]

        with MongoClient(**self.mongodb) as client:
            collection = client.cat_database[collection_name]
            try:
                collection.bulk_write(requests, ordered=False)
            except BulkWriteError as ex:
                # Duplicate keys are expected (and skipped) if `overwrite=False`
                write_errors = ex.details.get('writeErrors', [])
                other_error = any(err.get('code') != _DUPLICATE_KEY_CODE for err in write_errors)
                if other_error or ex.details.get('writeConcernErrors'):
                    raise

    def query_mongodb(
        self,
        database: Ligand | QD = 'ligand',
        query: Mapping[str | tuple[str, str], Any] | None = None,
        columns: Sequence[str | tuple[str, str]] | None = None,
        limit: int = 0,
    ) -> pd.DataFrame:
        """Read ligand or qd results from the MongoDB database into a DataFrame.

        The inverse of :meth:`Database.update_mongodb`: the returned DataFrame has the
        same layout as the DataFrames of :attr:`Database.csv_lig` and :attr:`Database.csv_qd`.

        Examples
        --------
        .. code:: python

            >>> from dataCAT import Database

            >>> db = Database(...)  # doctest: +SKIP

            # All ligands with a given SMILES string
            >>> db.query_mongodb('ligand', {'smiles': 'CC(=O)[O-]'})  # doctest: +SKIP

            # The bulkiness of all quantum dots with a given core
            >>> query = {'core': 'Cd68Cl26Se55'}
            >>> db.query_mongodb('qd', query, columns=['V_bulk'])  # doctest: +SKIP

            # Ligands with a cone angle below 120 degrees; columns are given as
            # (index, sub index) tuples and the values are MongoDB query operators
            >>> query = {('cone_angle', 'dist=0.0'): {'$lt': 120}}
            >>> db.query_mongodb('ligand', query)  # doctest: +SKIP

        Parameters
        ----------
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        query : :class:`Mapping<typing.Mapping>`, optional
            A MongoDB query filter (see :meth:`pymongo.collection.Collection.find`).
            Its keys can be index names (*e.g.* ``"smiles"`` or ``"core"``) or
            ``(index, sub index)`` tuples of columns, also within ``"$and"``,
            ``"$or"`` and ``"$nor"``.
            Column tuples whose sub index is empty or contains a ``"."`` need MongoDB >= 5.0.
        columns : :class:`Sequence<typing.Sequence>`, optional
            The columns to return, as ``index`` strings (all columns with that name) or
            ``(index, sub index)`` tuples; :data:`None` returns all columns.
        limit : :class:`int`
            The maximum number of documents to return; ``0`` means no limit.

        Returns
        -------
        :class:`pandas.DataFrame`
            A DataFrame with the matching documents, in the order of the collection.
            Values absent from a document are set to ``NaN``.

        """
        if self.mongodb is None:
            raise ValueError('Database.Mongodb is None')
        try:
            collection_name, idx_keys = MONGO_COLLECTIONS[database]
        except (KeyError, TypeError):
            raise ValueError(
                f"database={database!r}; accepted values for are 'ligand' and 'qd'"
            ) from None

        # Skip the placeholder row that `Database.update_mongodb` exports from the .csv file
        placeholder = {'$nor': [dict.fromkeys(idx_keys, '-')]}
        add_fields: dict[str, Any] = {}
        match = _parse_mongo_query(query or {}, add_fields)

        pipeline: list[dict[str, Any]] = []
        if add_fields:
            pipeline.append({'$addFields': add_fields})
        pipeline.append({'$match': {'$and': [placeholder, match]} if match else placeholder})
        if columns is None:
            pipeline.append({'$project': {'_id': 0, **dict.fromkeys(add_fields, 0)}})
        else:
            keep = {i if isinstance(i, str) else i[0] for i in columns}
            pipeline.append({'$project': {'_id': 0, **dict.fromkeys([*idx_keys, *keep], 1)}})
        if limit:
            pipeline.append({'$limit': limit})

        with MongoClient(**self.mongodb) as client:
            docs = list(client.cat_database[collection_name].aggregate(pipeline))
        df = mongo_dict_to_df(docs, idx_keys)
        if columns is None:
            return df.sort_index(axis=1)

        selection: list[tuple[str, str]] = []
        for i in columns:
            if isinstance(i, str):
                selection += sorted(j for j in df.columns if j[0] == i)
            else:
                selection.append(tuple(i))  # type: ignore[arg-type]
        return df.reindex(columns=pd.MultiIndex.from_tuples(selection, names=df.columns.names))

    @_locked
    def update_csv(self, df: pd.DataFrame,
                   index: None | slice | pd.Series = None,
                   database: Ligand | QD = 'ligand',
                   columns: Sequence | None = None,
                   overwrite: bool = False,
                   job_recipe: None = None,
                   status: str | None = None) -> None:
        """Update :attr:`Database.csv_lig` or :attr:`Database.csv_qd` with new settings.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` (:attr:`Database.csv_lig`)
            and ``"qd"`` (:attr:`Database.csv_qd`).
        columns : :class:`~collections.abc.Sequence`, optional
            Optional: A sequence of column keys in **df** which
            (potentially) are to be added to this instance.
            If :data:`None` Add all columns.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.
        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.


        :rtype: :data:`None`

        """
        # Operate on either the ligand or quantum dot database
        manager = self._parse_database(database)

        if job_recipe is not None:
            warnings.warn(
                "job_recipe .yaml storage has been discontinued", DeprecationWarning, stacklevel=2
            )

        df_index = slice(None) if index is None else index

        with manager(write=True) as db:
            # Update **db.index**
            db.ndframe = even_index(db.ndframe, df.loc[df_index])

            # Filter columns
            export_mol = False
            if columns is None:
                df_columns = df.columns
            else:
                df_columns = pd.Index(columns)
                if MOL in df_columns:
                    df_columns = df_columns.drop(MOL)
                    export_mol = True

            # Remove columns with the (now deprecated) `settings` key
            if isinstance(df_columns, pd.MultiIndex):
                df_columns = pd.Index([(i, j) for i, j in df_columns if i != 'settings'])

            # Update **db.columns**
            bool_ar = df_columns.isin(db.columns)
            df_columns = self._even_df_columns(df, db, df_columns, ~bool_ar)

            # Update **self.hdf5**; returns a new series of indices
            if export_mol:
                hdf5_series = self.update_hdf5(
                    df, df_index, database=database, overwrite=overwrite, status=status
                )

            # Update **db.values**
            # Use `loc`-based assignments for the integer and boolean columns, as
            # `DataFrame.update()` can upcast them to float or object (pandas < 2.0)
            db.update(df.loc[df_index, df_columns], overwrite=overwrite)
            if export_mol:
                db.loc[hdf5_series.index, HDF5_INDEX] = hdf5_series.to_numpy()
                if status == 'optimized':
                    opt_series = df.loc[df_index, OPT]
                    db.loc[opt_series.index, OPT] = opt_series.to_numpy()

        # Update the hdf5 file
        with self.hdf5('r+') as f:
            # Get the appropiate group
            name = 'ligand' if manager == self.csv_lig else 'qd'
            group = f[f'{name}/properties']

            # "Fix" for rare bug wherein the index dtype is incorrect
            if df[HDF5_INDEX].dtype.kind != "i":
                warnings.warn(f"Invalid {HDF5_INDEX!r} dtype: {df[HDF5_INDEX].dtype}", stacklevel=2)
                df[HDF5_INDEX] = _to_int64_array(df[HDF5_INDEX].values)

            # Define the indices; h5py requires them to be sorted
            df_sorted = df.sort_values(by=[HDF5_INDEX])
            hdf5_index = df_sorted[HDF5_INDEX].values

            # Define the properties
            lvl0_ignore = {OPT[0], HDF5_INDEX[0], MOL[0]}
            lvl0_ignore.update(i for i in df_columns.levels[0] if i.startswith("job_settings"))
            lvl0 = set(df_columns.levels[0]).difference(lvl0_ignore)

            dct = {k: df_columns.get_loc_level(k)[1] for k in lvl0}
            for n, name_seq in dct.items():
                data = df_sorted[n].to_numpy()
                dtype = h5py.string_dtype(encoding='utf-8') if data.dtype == object else data.dtype

                # Get the dataset
                try:
                    dset = group[n]
                except KeyError:
                    if not name_seq.any():
                        name_seq = None
                    dset = create_prop_dset(group, n, dtype, name_seq)

                # Update the dataset
                update_prop_dset(dset, data, hdf5_index)

            # Add an entry to the logger
            names = [group[k].name for k in dct]
            message = f'datasets={names!r}; overwrite={overwrite!r}'
            update_hdf5_log(f[f'{name}/logger'], hdf5_index, message=message)

    def _even_df_columns(self, df: pd.DataFrame, db: DFProxy,
                         columns: MIT, subset: np.ndarray) -> MIT:
        """Even the columns of **df** and **db**."""
        drop_idx = []
        for i in columns[subset]:
            # Check for job settings
            if 'job_settings' in i[0]:
                self._update_hdf5_settings(df, i[0])
                del df[i]
                drop_idx.append(i)
                continue

            # Ensure that **db** has the same keys as **df**
            dtype = df[i].dtype
            if isinstance(dtype, pd.api.extensions.ExtensionDtype):
                db[i] = pd.Series(dtype.na_value, index=db.index, dtype=dtype)
                continue
            try:
                db[i] = np.array((None), dtype=dtype)
            except TypeError:  # e.g. if csv[i] consists of the datatype np.int64
                db[i] = -1
        return columns.drop(drop_idx)  # type: ignore

    @_locked
    def update_hdf5(self, df: pd.DataFrame,
                    index: slice | pd.Series,
                    database: Ligand | QD = 'ligand',
                    overwrite: bool = False,
                    status: str | None = None) -> pd.Series:
        """Export molecules (see the ``"mol"`` column in **df**) to the structure database.

        Returns a series with the :attr:`Database.hdf5` indices of all new entries.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.
        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.

        Returns
        -------
        :class:`pandas.Series`
            A series with the indices of all new molecules in :attr:`Database.hdf5`.

        """
        # Identify new and preexisting entries
        sub_df = df.loc[index, :]
        if status == 'optimized':
            new = sub_df.loc[(sub_df[OPT] == False) & ~sub_df[MOL].isnull(), HDF5_INDEX]  # noqa
            old = sub_df.loc[sub_df[OPT] == True, HDF5_INDEX]  # noqa
            opt = True
        else:
            new = sub_df.loc[(sub_df[HDF5_INDEX] == -1) & ~sub_df[MOL].isnull(), HDF5_INDEX]
            old = sub_df.loc[sub_df[HDF5_INDEX] >= 0, HDF5_INDEX]
            opt = False

        # Add new entries to the database
        self.hdf5_availability()
        with self.hdf5('r+') as f:
            group = f[database]
            dtype = IDX_DTYPE[database]

            if len(new):
                ret = self._write_hdf5(group, df, new.index, dtype, database, opt=opt)
            else:
                ret = pd.Series(name=HDF5_INDEX, dtype=int)

            # If **overwrite** is *True*
            if overwrite and old.any():
                self._overwrite_hdf5(group, old, df, dtype, opt=opt)
        return ret

    @classmethod
    def _write_hdf5(cls, group: h5py.Group, df: pd.DataFrame, new_index: pd.Index,
                    dtype: DTypeLike, database: Ligand | QD = 'ligand',
                    opt: bool = False) -> pd.Index:
        """Helper method for :meth:`update_hdf5` when :code:`overwrite = False`."""
        mol_series = df.loc[new_index, MOL]

        i = len(group['atoms'])
        j = i + len(mol_series)
        ret = pd.Series(np.arange(i, j), index=new_index, name=HDF5_INDEX)

        scale = cls._sanitize_multi_idx(new_index, dtype, database)
        pdb_new = PDBContainer.from_molecules(mol_series, scale=scale)
        pdb_new.to_hdf5(group, index=np.s_[i:j], update_scale=not opt)

        names = ('atoms', 'bonds', 'atom_count', 'bond_count')
        message = f"datasets={[group[n].name for n in names]!r}; overwrite=False"
        update_hdf5_log(group['logger'], index=ret.values, message=message)
        df.loc[new_index, HDF5_INDEX] = ret.to_numpy()
        if opt:
            df.loc[new_index, OPT] = True
        return ret

    @staticmethod
    def _overwrite_hdf5(group: h5py.Group, old: pd.Series, df: pd.DataFrame,
                        dtype: DTypeLike, opt: bool = False) -> None:
        """Helper method for :meth:`update_hdf5` when :code:`overwrite = True`."""
        mol_series = df.loc[old.index, MOL]

        scale = mol_series.index.values.astype(dtype)
        pdb_old = PDBContainer.from_molecules(mol_series, scale=scale)
        pdb_old.to_hdf5(group, index=old.values)

        names = ('atoms', 'bonds', 'atom_count', 'bond_count')
        message = f"datasets={[group[n].name for n in names]!r}; overwrite=True"
        update_hdf5_log(group['logger'], index=old.values, message=message)
        if opt:
            df.loc[old.index, OPT] = True

    @staticmethod
    def _sanitize_multi_idx(index: MIT, dtype: DTypeLike, database: Ligand | QD) -> MIT:
        """Parse and sanitize the passed MultiIndex."""
        return index.values.astype(dtype)  # type: ignore

    def _update_hdf5_settings(self, df: pd.DataFrame, column: str) -> None:
        """Export all files in **df[column]** to hdf5 dataset **column**."""
        self.hdf5_availability()
        with self.hdf5('r+') as f:
            _update_hdf5_settings(f, df, column)

    """ ########################  Pulling results from the database ########################### """

    @_locked
    def from_csv(self, df: pd.DataFrame, database: Ligand | QD = 'ligand',
                 get_mol: bool = True, inplace: bool = True) -> pd.Series | None:
        """Pull results from :attr:`Database.csv_lig` or :attr:`Database.csv_qd`.

        Performs in inplace update of **df** if **inplace** = :data:`True`,
        thus returing :data:`None`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        get_mol : :class:`bool`
            Attempt to pull preexisting molecules from the database.
            See the **inplace** argument for more details.
        inplace : :class:`bool`
            If :data:`True` perform an inplace update of the ``"mol"`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        :class:`pandas.Series`, optional
            Optional: A Series of PLAMS molecules if **get_mol** = :data:`True`
            and **inplace** = :data:`False`.

        """
        # Operate on either the ligand or quantum dot database
        manager = self._parse_database(database)

        # Update the *hdf5 index* column in **df**
        with manager(write=False) as db:
            df.update(db.ndframe, overwrite=True)
            df[HDF5_INDEX] = df[HDF5_INDEX].astype(int)

        # **df** has been updated and **get_mol** = *False*
        if not get_mol:
            return None
        return self._get_csv_mol(df, database, inplace)

    def _get_csv_mol(self, df: pd.DataFrame,
                     database: Ligand | QD = 'ligand',
                     inplace: bool = True) -> pd.Series | None:
        """A method which handles the retrieval and subsequent formatting of molecules.

        Called internally by :meth:`Database.from_csv`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        inplace : :class:`bool`
            If :data:`True` perform an inplace update of the ``("mol", "")`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        :class:`pandas.Series`, optional
            Optional: A Series of PLAMS molecules if **inplace** is :data:`False`.

        """
        # Sort and find all valid HDF5 indices; h5py requires them to be sorted
        df_sorted = df.sort_values(by=[HDF5_INDEX])
        if 'no_opt' in database:  # noqa: SIM108
            df_slice = df_sorted[HDF5_INDEX] >= 0
        else:
            df_slice = df_sorted[OPT] == True  # noqa
        idx = df_sorted[HDF5_INDEX][df_slice].values

        # If no HDF5 indices are availble in **df** then abort the function
        if not df_slice.any():
            if inplace:
                return None
            return pd.Series(None, name=MOL, dtype=object)

        # Update **df** with preexisting molecules from **self**, returning *None*
        if inplace:
            mol_list = df_sorted.loc[df_slice, MOL]
            self.from_hdf5(idx, database=database, mol_list=mol_list, rdmol=False)
            return None

        # Create and return a new series of PLAMS molecules
        else:
            mol_list = self.from_hdf5(idx, database=database, rdmol=False)
            return pd.Series(mol_list, index=df_sorted[df_slice].index, name=MOL)

    @_locked
    def from_hdf5(self, index: slice | Sequence[int],
                  database: Ligand | QD = 'ligand', rdmol: bool = True,
                  mol_list: Iterable[Molecule] | None = None) -> list[Molecule | Mol]:
        """Import structures from the hdf5 database as RDKit or PLAMS molecules.

        Parameters
        ----------
        index : :class:`Sequence[int]<typing.Sequence>` or :class:`slice`
            The indices of the to be retrieved structures.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        rdmol : :class:`bool`
            If :data:`True`, return an RDKit molecule instead of a PLAMS molecule.

        Returns
        -------
        :class:`List[plams.Molecule]<typing.List>` or :class:`List[rdkit.Mol]<typing.List>`
            A list of PLAMS or RDKit molecules.

        """
        # Open the database and pull entries
        self.hdf5_availability()
        with self.hdf5('r') as f:
            pdb = PDBContainer.from_hdf5(f[database], index)
            mol_list_ = pdb.to_molecules(mol=mol_list)

        is_opt = 'no_opt' not in database
        for mol in mol_list_:
            mol.properties.is_opt = is_opt

        if rdmol:
            return [to_rdmol(mol) for mol in mol_list_]
        return mol_list_

    def hdf5_availability(self, timeout: float = 5.0,
                          max_attempts: int | None = 10) -> None:
        """Check if a .hdf5 file is opened by another process; return once it is not.

        If two processes attempt to simultaneously open a single hdf5 file then
        h5py will raise an :exc:`OSError`.

        The purpose of this method is ensure that a .hdf5 file is actually closed,
        thus allowing the :meth:`Database.from_hdf5` method to safely access **filename** without
        the risk of raising an :exc:`OSError`.
        It first waits for the lock of the database (see :mod:`dataCAT.locking`), which is held by
        all other processes that use this version of Data-CAT or newer,
        and then checks whether a process without the lock has the file open.

        Parameters
        ----------
        timeout : :class:`float`
            Time timeout, in seconds, between subsequent attempts of opening **filename**.
        max_attempts : :class:`int`, optional
            Optional: The maximum number attempts for opening **filename**.
            If the maximum number of attempts is exceeded, raise an :exc:`OSError`.
            Setting this value to :data:`None` will set the number of attempts to unlimited.

        Raises
        ------
        :exc:`OSError`
            Raised if **max_attempts** is exceded.

        See Also
        --------
        :func:`dataCAT.functions.hdf5_availability`
            This method as a function.

        """
        filename = self.hdf5.args[0]
        with get_lock(self.dirname):
            hdf5_availability(filename, timeout, max_attempts, libver='latest')
