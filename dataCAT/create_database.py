"""A module for creating database files for the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.create_database
.. autosummary::
    create_csv
    create_hdf5
    create_mongodb

API
---
.. autofunction:: create_csv
.. autofunction:: create_hdf5
.. autofunction:: create_mongodb

"""

from os import PathLike
from os.path import join, isfile
from types import MappingProxyType
from logging import Logger
from typing import Any, Union, AnyStr, Literal, overload
from collections.abc import Mapping

import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING

from nanoutils import PathType
from CAT.logger import logger

from .dtype import BACKUP_IDX_DTYPE, LIG_IDX_DTYPE, QD_IDX_DTYPE, FORMULA_DTYPE, LIG_COUNT_DTYPE
from .hdf5_log import create_hdf5_log
from .pdb_array import PDBContainer
from .functions import from_pdb_array, _set_index
from .property_dset import create_prop_dset, create_prop_group

__all__ = ['create_csv', 'create_hdf5', 'create_mongodb']

Ligand = Literal['ligand', 'ligand_no_opt']
QD = Literal['qd', 'qd_no_opt']


def create_csv(path: str | PathLike, database: Ligand | QD = 'ligand') -> str:
    """Create a ligand or qd database (csv format) if it does not yet exist.

    Parameters
    ----------
    path : str
        The path (without filename) of the database.

    database : |str|_
        The type of database, accepted values are ``"ligand"`` and ``"qd"``.

    Returns
    -------
    |str|_
        The absolute path to the ligand or qd database.

    """
    filename = join(path, f'{database}_database.csv')

    # Check if the database exists and has the proper keys; create it if it does not
    if not isfile(filename):
        if database == 'ligand':
            _create_csv_lig(filename)
        elif database == 'qd':
            _create_csv_qd(filename)
        else:
            raise ValueError(f"{database!r} is not an accepated value for the 'database' argument")
        logger.info(f'{database}_database.csv not found in {path}, creating {database} database')
    return filename


def _create_csv_lig(filename: PathType) -> None:
    """Create a ligand database and and return its absolute path.

    Parameters
    ----------
    path : str
        The path+filename of the ligand database.

    """
    idx = pd.MultiIndex.from_tuples([('-', '-')], names=['smiles', 'anchor'])

    columns = pd.MultiIndex.from_tuples(
        [('hdf5 index', ''), ('formula', ''), ('opt', ''), ('settings', 1)],
        names=['index', 'sub index']
    )

    df = pd.DataFrame(None, index=idx, columns=columns)
    df['hdf5 index'] = -1
    df['formula'] = 'str'
    df['settings'] = 'str'
    df['opt'] = False
    df.to_csv(filename)


def _create_csv_qd(filename: PathType) -> None:
    """Create a qd database and and return its absolute path.

    Parameters
    ----------
    path : str
        The path+filename of the qd database.

    """
    idx = pd.MultiIndex.from_tuples(
        [('-', '-', '-', '-')],
        names=['core', 'core anchor', 'ligand smiles', 'ligand anchor']
    )

    columns = pd.MultiIndex.from_tuples(
        [('hdf5 index', ''), ('ligand count', ''), ('opt', ''), ('settings', 1), ('settings', 2)],
        names=['index', 'sub index']
    )

    df = pd.DataFrame(None, index=idx, columns=columns)
    df['hdf5 index'] = -1
    df['ligand count'] = -1
    df['settings'] = 'str'
    df['opt'] = False
    df.to_csv(filename)


IDX_DTYPE: Mapping[str, np.dtype] = MappingProxyType({
    'core': BACKUP_IDX_DTYPE,
    'core_no_opt': BACKUP_IDX_DTYPE,
    'ligand': LIG_IDX_DTYPE,
    'ligand_no_opt': LIG_IDX_DTYPE,
    'qd': QD_IDX_DTYPE,
    'qd_no_opt': QD_IDX_DTYPE
})


#: The names of the MongoDB collections and their (unique) index keys.
MONGO_COLLECTIONS: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType({
    'ligand': ('ligand_database', ('smiles', 'anchor')),
    'qd': ('qd_database', ('core', 'core anchor', 'ligand smiles', 'ligand anchor')),
})


DEFAULT_PROPERTIES: Mapping[str, tuple[str, np.dtype] | None] = MappingProxyType({
    'core': None,
    'core_no_opt': None,
    'ligand': ('formula', FORMULA_DTYPE),
    'ligand_no_opt': None,
    'qd': ('ligand count', LIG_COUNT_DTYPE),
    'qd_no_opt': None
})


@overload
def create_hdf5(path: Union[AnyStr, 'PathLike[AnyStr]']) -> AnyStr:
    ...
@overload  # noqa: E302
def create_hdf5(path: Union[AnyStr, 'PathLike[AnyStr]'], name: AnyStr) -> AnyStr:
    ...
def create_hdf5(path, name='structures.hdf5'):  # noqa: E302
    """Create the .pdb structure database (hdf5 format).

    Parameters
    ----------
    path : str
        The path (without filename) to the database.

    name : str
        The filename of the database (excluding its path).

    Returns
    -------
    |str|_
        The absolute path+filename to the pdb structure database.

    """
    path = join(path, name)

    # Define arguments for 2D datasets
    dataset_names = ('core', 'core_no_opt', 'ligand', 'ligand_no_opt', 'qd', 'qd_no_opt')
    scale_dict = {'core_no_opt': 'core', 'ligand_no_opt': 'ligand', 'qd_no_opt': 'qd'}
    kwargs = {'chunks': True, 'compression': 'gzip'}

    # Define arguments for 3D datasets
    kwargs_3d = {'chunks': True, 'maxshape': (None, None, None), 'compression': 'gzip'}
    dataset_names_3d = (
        'job_settings_crs', 'job_settings_qd_opt', 'job_settings_BDE', 'job_settings_ASA',
        'job_settings_cdft'
    )

    with h5py.File(path, 'a', libver='latest') as f:
        for grp_name in dataset_names:
            # Check for pre-dataCAT-0.3 style databases
            pdb = _update_pdb_dsets(f, grp_name, logger)

            # Create a new group if it does not exist yet
            if grp_name not in f:
                idx_grp = scale_dict.get(grp_name)
                if idx_grp is not None:
                    scale = f[f'{idx_grp}/{PDBContainer.SCALE_NAME}']
                    group = PDBContainer.create_hdf5_group(f, grp_name, scale=scale, **kwargs)
                else:
                    dtype = IDX_DTYPE[grp_name]
                    group = PDBContainer.create_hdf5_group(f, grp_name, scale_dtype=dtype, **kwargs)  # noqa: E501
            else:
                group = f[grp_name]

            # Check for pre-dataCAT-0.4 style databases
            _update_index_dset(group, grp_name, logger)

            if pdb is not None:
                pdb.to_hdf5(group, None)

            _update_property_dsets(group, grp_name)

            # Check of the log is present
            if 'logger' not in group:
                create_hdf5_log(group, compression='gzip')

        # Create new 3D datasets
        iterator_3d = (grp_name for grp_name in dataset_names_3d if grp_name not in f)
        for grp_name in iterator_3d:
            f.create_dataset(grp_name, data=np.empty((0, 1, 1), dtype='S120'), **kwargs_3d)

    return path


def _update_pdb_dsets(file: h5py.File, name: str,
                      logger: Logger | None = None) -> PDBContainer | None:
    """Check for and update pre dataCAT 0.3 style databases."""
    if not isinstance(file.get(name), h5py.Dataset):
        return None
    elif logger is not None:
        logger.info(f'Updating h5py Dataset to data-CAT >= 0.3 style: {name!r}')

    mol_list = [from_pdb_array(pdb, rdmol=False, warn=False) for pdb in file[name]]
    m = len(mol_list)
    del file[name]

    dtype = IDX_DTYPE[name]
    scale = np.rec.array(None, dtype=dtype, shape=(m,))
    if dtype.fields is not None and scale.size:
        # Ensure that the sentinal value for vlen strings is an empty string, not `None`
        elem = list(scale.item(0))
        iterator = (v for v, *_ in dtype.fields.values())
        for i, sub_dt in enumerate(iterator):
            if h5py.check_string_dtype(sub_dt) is not None:
                elem[i] = ''
        scale[:] = tuple(elem)
    return PDBContainer.from_molecules(mol_list, scale=scale)


def _update_index_dset(group: h5py.Group, name: str, logger: Logger | None = None) -> None:
    """Check for and update pre dataCAT 0.4 style databases."""
    if 'index' in group:
        return None
    elif logger is not None:
        logger.info(f'Updating h5py Dataset to data-CAT >= 0.4 style: {name!r}')

    dtype = IDX_DTYPE[name]
    i = len(group['atoms'])
    _set_index(PDBContainer, group, dtype, i, compression='gzip')


def _update_property_dsets(group: h5py.Group, name: str) -> None:
    """Check for and update pre dataCAT 0.4 style databases."""
    if 'properties' in group:
        return None

    scale = group['index']
    prop_grp = create_prop_group(group, scale)

    args = DEFAULT_PROPERTIES[name]
    if args is not None:
        create_prop_dset(prop_grp, *args, compression='gzip')


def create_mongodb(host: str = 'localhost', port: int = 27017, **kwargs: Any) -> dict[str, Any]:
    """Create the the MongoDB collections and set their index.

    Paramaters
    ----------
    host : :class:`str`
        Hostname or IP address or Unix domain socket path of a single mongod or
        mongos instance to connect to, or a mongodb URI, or a list of hostnames mongodb URIs.
        If **host** is an IPv6 literal it must be enclosed in ``"["`` and ``"]"`` characters
        following the RFC2732 URL syntax (e.g. ``"[::1]"`` for localhost).
        Multihomed and round robin DNS addresses are not supported.

    port : :class:`int`
        port number on which to connect.

    kwargs : :data:`Any<typing.Any>`
        Optional keyword argument for :class:`pymongo.MongoClient`.

    Returns
    -------
    :class:`Dict[str, Any]<typing.Dict>`
        A dictionary with all supplied keyword arguments.

    Raises
    ------
    :exc:`pymongo.ServerSelectionTimeoutError`
        Raised if no connection can be established with the host.

    """  # noqa
    # Open the client
    client_kwargs = {'serverSelectionTimeoutMS': 1000, **kwargs}
    with MongoClient(host, port, **client_kwargs) as client:
        # Raises an ServerSelectionTimeoutError error if the server is inaccesible
        client.server_info()

        # Open the database
        db = client.cat_database

        # Open the collections and set their (unique) indices
        for collection_name, idx_keys in MONGO_COLLECTIONS.values():
            collection = db[collection_name]
            idx_name = '_'.join(f'{k}_1' for k in idx_keys)
            if idx_name not in collection.index_information():
                collection.create_index([(k, ASCENDING) for k in idx_keys], unique=True)

    # Return all provided keyword argument
    ret = {'host': host, 'port': port}
    ret.update(kwargs)
    return ret
