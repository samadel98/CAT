"""Tests for the :class:`dataCAT.Database` class."""

import os
import sys
import copy
import shutil
import pickle
from types import MappingProxyType
from os.path import join, abspath
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from pymongo import MongoClient

from nanoutils import delete_finally
from assertionlib import assertion
from CAT.utils import get_formula
from CAT.workflows import MOL, HDF5_INDEX, OPT, V_BULK, JOB_SETTINGS_CDFT

from dataCAT import Database, OpenLig, OpenQD
from dataCAT.locking import open_hdf5
from dataCAT.create_database import MONGO_COLLECTIONS
from dataCAT.functions import df_to_mongo_dict, mongo_dict_to_df
from dataCAT.testing_utils import _readpdb as readpdb

PATH = Path('tests') / 'test_files'
DB_PATH = PATH / 'database'
DB_PATH_UPDATE = PATH / 'database_update'

DB = Database(DB_PATH)


def test_init() -> None:
    """Test :meth:`dataCAT.Database.__init__`."""
    assertion.eq(DB.dirname, abspath(DB_PATH))

    assertion.eq(DB.csv_lig.keywords['filename'], abspath(join(DB_PATH, 'ligand_database.csv')))
    assertion.is_(DB.csv_lig.func, OpenLig)

    assertion.eq(DB.csv_qd.keywords['filename'], abspath(join(DB_PATH, 'qd_database.csv')))
    assertion.is_(DB.csv_qd.func, OpenQD)

    assertion.eq(DB.hdf5.args[0], abspath(join(DB_PATH, 'structures.hdf5')))
    assertion.is_(DB.hdf5.func, open_hdf5)

    assertion.isinstance(DB.mongodb, (type(None), MappingProxyType))


def test_eq() -> None:
    """Test :meth:`dataCAT.Database.__eq__`."""
    db2 = Database(DB_PATH)
    assertion.eq(db2, DB)
    assertion.eq(hash(db2), hash(DB))

    assertion.is_(DB, copy.copy(DB))
    assertion.is_(DB, copy.deepcopy(DB))

    dump = pickle.dumps(DB)
    db3 = pickle.loads(dump)
    assertion.eq(db3, DB)

    db_str = repr(DB)
    assertion.contains(db_str, DB.__class__.__name__)
    if sys.platform != 'win32':
        for name in ('dirname', 'csv_lig', 'csv_qd', 'hdf5'):
            assertion.contains(db_str, str(getattr(DB, name)), message=name)


def test_parse_database() -> None:
    """Test :meth:`dataCAT.Database._parse_database`."""
    out1 = DB._parse_database('ligand')
    out2 = DB._parse_database('qd')

    assertion.is_(out1, DB.csv_lig)
    assertion.is_(out2, DB.csv_qd)

    assertion.assert_(DB._parse_database, 'bob', exception=ValueError)


def test_hdf5_availability() -> None:
    """Test :meth:`dataCAT.Database.hdf5_availability`."""
    with DB.hdf5('r'):
        assertion.assert_(DB.hdf5_availability, 1.0, 2, exception=OSError)


def test_from_hdf5() -> None:
    """Test :meth:`dataCAT.Database.from_hdf5`."""
    ref_tup = ('C1H3O1', 'C2H5O1', 'C3H7O1')

    idx1 = slice(0, None)
    mol_list1 = DB.from_hdf5(idx1, 'ligand', rdmol=False)
    for mol, ref in zip(mol_list1, ref_tup):
        assertion.eq(get_formula(mol), ref)

    idx2 = np.arange(0, 2)
    mol_list2 = DB.from_hdf5(idx2, 'ligand', rdmol=False)
    for mol, ref in zip(mol_list2, ref_tup[0:2]):
        assertion.eq(get_formula(mol), ref)

    rdmol_list = DB.from_hdf5(idx1, 'ligand', rdmol=True)
    assertion.len_eq(rdmol_list, len(ref_tup))
    for rdmol, mol in zip(rdmol_list, mol_list1):
        assertion.isinstance(rdmol, Chem.Mol)
        assertion.eq(rdmol.GetNumAtoms(), len(mol.atoms))


def test_from_csv() -> None:
    """Test :meth:`dataCAT.Database.from_csv`."""
    columns = pd.MultiIndex.from_tuples(
        [('mol', ''), ('hdf5 index', ''), ('opt', '')]
    )
    idx = pd.MultiIndex.from_tuples(
        [('C[O-]', 'O2'), ('CC[O-]', 'O3'), ('CCC[O-]', 'O4')]
    )
    df = pd.DataFrame(np.nan, index=idx, columns=columns)

    df[MOL] = None
    df[HDF5_INDEX] = np.arange(0, 3, dtype=int)
    df[OPT] = True

    out1 = DB.from_csv(df, 'ligand', get_mol=False)
    assertion.is_(out1, None)

    ref_tup = ('C1H3O1', 'C2H5O1', 'C3H7O1')
    out2: pd.Series = DB.from_csv(df, 'ligand', get_mol=True, inplace=False)
    for mol, ref in zip(out2, ref_tup):
        assertion.eq(get_formula(mol), ref)


@delete_finally(DB_PATH_UPDATE)
def test_update_hdf5() -> None:
    """Test :meth:`~dataCAT.Database.update_hdf5`."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)

    mol_dict = {
        ('CC(=O)[O-]', 'O4'): readpdb(str(PATH / 'CC[=O][O-]@O4.pdb')),
        ('CCC(=O)[O-]', 'O5'): readpdb(str(PATH / 'CCC[=O][O-]@O5.pdb')),
    }
    df = pd.DataFrame({MOL: mol_dict})
    df[HDF5_INDEX] = -1
    df[OPT] = False

    series1 = db.update_hdf5(df, np.s_[:], database='ligand')
    assertion.eq(df[HDF5_INDEX], [3, 4], post_process=np.all)
    assertion.eq(series1, [3, 4], post_process=np.all)

    series2 = db.update_hdf5(df, np.s_[:], database='ligand')
    assertion.eq(df[HDF5_INDEX], [3, 4], post_process=np.all)
    assertion.eq(series2, [], post_process=np.all)

    series3 = db.update_hdf5(df, np.s_[:], database='ligand', overwrite=True)
    assertion.eq(df[HDF5_INDEX], [3, 4], post_process=np.all)
    assertion.eq(series3, [], post_process=np.all)


@delete_finally(DB_PATH_UPDATE)
def test_update_csv() -> None:
    """Test :meth:`~dataCAT.Database.update_csv`."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)

    mol_dict = {
        ('CC(=O)[O-]', 'O4'): readpdb(str(PATH / 'CC[=O][O-]@O4.pdb')),
        ('CCC(=O)[O-]', 'O5'): readpdb(str(PATH / 'CCC[=O][O-]@O5.pdb')),
    }
    df = pd.DataFrame({MOL: mol_dict})
    df[HDF5_INDEX] = [0, 1]
    df[OPT] = False
    df[V_BULK] = np.arange(2, dtype=float)

    db.update_csv(df, database='ligand', columns=[V_BULK])
    with db.csv_lig(write=False) as df2:
        np.testing.assert_allclose(df[V_BULK], df2.loc[df.index, V_BULK])

    db.update_csv(df, database='ligand', columns=[V_BULK])
    with db.csv_lig(write=False) as df2:
        np.testing.assert_allclose(df[V_BULK], df2.loc[df.index, V_BULK])

    df[V_BULK] *= 100
    db.update_csv(df, database='ligand', columns=[V_BULK], overwrite=True)
    with db.csv_lig(write=False) as df2:
        np.testing.assert_allclose(df[V_BULK], df2.loc[df.index, V_BULK])

    with db.hdf5('r') as f:
        dset = f['ligand/properties/V_bulk']
        idx = df[HDF5_INDEX].values
        np.testing.assert_array_equal(dset[idx], df[V_BULK])

        log = f['ligand/logger']
        msg = b"datasets=['/ligand/properties/V_bulk']; overwrite=False"
        assertion.eq(log['message'][3], msg)
        np.testing.assert_array_equal(log['index'][3], idx)


@delete_finally(DB_PATH_UPDATE)
def test_update_csv_new_and_old() -> None:
    """Test :meth:`~dataCAT.Database.update_csv` with a mix of new and pre-existing structures."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)

    idx = pd.MultiIndex.from_tuples([('CC(=O)[O-]', 'O4'), ('C[O-]', 'O2'), ('CC[O-]', 'O3')])
    mol_list = [
        readpdb(PATH / 'CC[=O][O-]@O4.pdb'),
        readpdb(PATH / 'CC[=O][O-]@O4.pdb'),
        readpdb(PATH / 'CCC[=O][O-]@O5.pdb'),
    ]
    df = pd.DataFrame({MOL: mol_list}, index=idx)
    df[HDF5_INDEX] = [-1, 0, 1]
    df[OPT] = False
    df[V_BULK] = np.arange(3, dtype=float)
    df_index = df.index.copy()

    # The new structure is assigned the largest hdf5 index (`3`), thus requiring the
    # rows of `df` to be sorted (again) before exporting the properties to the hdf5 file
    db.update_csv(df, database='ligand', columns=[MOL, V_BULK])
    np.testing.assert_array_equal(df.index, df_index)
    np.testing.assert_array_equal(df[HDF5_INDEX], [3, 0, 1])

    with db.csv_lig(write=False) as df2:
        np.testing.assert_allclose(df[V_BULK], df2.loc[df.index, V_BULK])
        np.testing.assert_array_equal(df[HDF5_INDEX], df2.loc[df.index, HDF5_INDEX])

    with db.hdf5('r') as f:
        dset = f['ligand/properties/V_bulk']
        np.testing.assert_array_equal(dset[[0, 1, 3]], [1.0, 2.0, 0.0])
        assertion.len_eq(f['ligand/atoms'], 4)


@delete_finally(DB_PATH_UPDATE)
def test_update_csv_str() -> None:
    """Test :meth:`~dataCAT.Database.update_csv` with a new column of strings."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)

    column = ('name', '')
    idx = pd.MultiIndex.from_tuples([('C[O-]', 'O2'), ('CC[O-]', 'O3')])
    df = pd.DataFrame({HDF5_INDEX: [0, 1], OPT: True, column: ['methoxide', 'ethoxide']}, index=idx)

    db.update_csv(df, database='ligand', columns=[column])
    with db.csv_lig(write=False) as df2:
        assertion.eq(df2.loc[idx, column].tolist(), ['methoxide', 'ethoxide'])
        assertion.truth(pd.isna(df2.loc[('CCC[O-]', 'O4'), column]))

    with db.hdf5('r') as f:
        dset = f['ligand/properties/name']
        assertion.eq(dset[:2].astype(str).tolist(), ['methoxide', 'ethoxide'])


@delete_finally(DB_PATH_UPDATE)
def test_update_hdf5_settinga() -> None:
    """Test :meth:`~dataCAT.Database._update_hdf5_settings`."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)
    job_settings = [
        PATH / 'CDFT' / 'CDFT' / 'CDFT.in',
        PATH / 'CDFT' / 'CDFT.002' / 'CDFT.002.in',
        PATH / 'CDFT' / 'CDFT.003' / 'CDFT.003.in',
        PATH / 'CDFT' / 'CDFT.004' / 'CDFT.004.in'
    ]

    mol_dict = {
        ('CC(=O)[O-]', 'O4'): readpdb(str(PATH / 'CC[=O][O-]@O4.pdb'))
    }
    df = pd.DataFrame({MOL: mol_dict})
    df[HDF5_INDEX] = [3]
    df[JOB_SETTINGS_CDFT] = [job_settings]

    db._update_hdf5_settings(df, 'job_settings_cdft')
    with db.hdf5('r') as f:
        job_ar = f['job_settings_cdft'][-1].astype(str)

    for file, ar in zip(job_settings, job_ar):
        with open(file, 'r') as f:
            try:
                iterator = (i.rstrip('\n') for i in f)
                for i, j in zip(iterator, ar):
                    while not i:
                        i = next(iterator)
                    assertion.eq(i, j)
            except StopIteration:
                pass


#: A prefix for all MongoDB document keys created by the tests below,
#: ensuring that only documents created by the tests are removed afterwards.
MONGO_PREFIX = 'dataCAT-test:'

#: The names of the MongoDB collections and the keys of their first index.
MONGO_KEYS = MappingProxyType({
    'ligand_database': 'smiles',
    'qd_database': 'core',
})


def _clear_mongodb(db: Database) -> None:
    """Remove all MongoDB documents with :data:`MONGO_PREFIX`-prefixed keys."""
    with MongoClient(**db.mongodb) as client:
        for name, key in MONGO_KEYS.items():
            client.cat_database[name].delete_many({key: {'$regex': f'^{MONGO_PREFIX}'}})


@pytest.fixture
def mongo_db():
    """Yield a copy of the test database whose index keys have been prefixed with :data:`MONGO_PREFIX`."""  # noqa: E501
    if DB.mongodb is None:
        # Ensure that the tests are not silently skipped in CI
        if os.environ.get('DATACAT_REQUIRE_MONGODB'):
            pytest.fail("MongoDB server not found")
        pytest.skip("MongoDB server not found")

    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)
    for manager in (db.csv_lig, db.csv_qd):
        with manager(write=True) as df:
            df.index = df.index.map(lambda tup: tuple(f'{MONGO_PREFIX}{i}' for i in tup))

    _clear_mongodb(db)
    try:
        yield db
    finally:
        _clear_mongodb(db)
        shutil.rmtree(DB_PATH_UPDATE)


def _get_mongo_docs(db: Database, name: str) -> dict:
    """Return all :data:`MONGO_PREFIX`-prefixed documents of the **name** collection."""
    key = MONGO_KEYS[name]
    with MongoClient(**db.mongodb) as client:
        collection = client.cat_database[name]
        cursor = collection.find({key: {'$regex': f'^{MONGO_PREFIX}'}}, {'_id': False})
        return {doc[key]: doc for doc in cursor}


def test_update_mongodb_ligand(mongo_db: Database) -> None:
    """Test :meth:`~dataCAT.Database.update_mongodb` with the ligand database."""
    with mongo_db.csv_lig(write=False) as df:
        n = len(df)

    # The second call should silently skip all (now pre-existing) documents
    mongo_db.update_mongodb('ligand')
    mongo_db.update_mongodb('ligand')

    docs = _get_mongo_docs(mongo_db, 'ligand_database')
    assertion.len_eq(docs, n)

    doc = docs[f'{MONGO_PREFIX}CC[O-]']
    assertion.eq(doc['anchor'], f'{MONGO_PREFIX}O3')
    assertion.eq(doc['formula'], {'': 'C2H5O1'})
    assertion.eq(doc['hdf5 index'], {'': 1})
    assertion.is_(doc['opt'][''], True)


def test_update_mongodb_overwrite(mongo_db: Database) -> None:
    """Test :meth:`~dataCAT.Database.update_mongodb` with :code:`overwrite=True`."""
    smiles = f'{MONGO_PREFIX}CC[O-]'
    mongo_db.update_mongodb('ligand')

    with mongo_db.csv_lig(write=True) as df:
        df.loc[(smiles, f'{MONGO_PREFIX}O3'), ('formula', '')] = 'bob'

    mongo_db.update_mongodb('ligand', overwrite=False)
    doc = _get_mongo_docs(mongo_db, 'ligand_database')[smiles]
    assertion.eq(doc['formula'], {'': 'C2H5O1'})

    mongo_db.update_mongodb('ligand', overwrite=True)
    docs = _get_mongo_docs(mongo_db, 'ligand_database')
    assertion.eq(docs[smiles]['formula'], {'': 'bob'})
    with mongo_db.csv_lig(write=False) as df:
        assertion.len_eq(docs, len(df))


def test_update_mongodb_qd(mongo_db: Database) -> None:
    """Test :meth:`~dataCAT.Database.update_mongodb` with the QD database."""
    with mongo_db.csv_qd(write=False) as df:
        n = len(df)

    mongo_db.update_mongodb('qd')
    mongo_db.update_mongodb('qd')

    docs = _get_mongo_docs(mongo_db, 'qd_database')
    assertion.len_eq(docs, 2)  # The index of all non-placeholder rows starts with the same core
    with MongoClient(**mongo_db.mongodb) as client:
        collection = client.cat_database.qd_database
        query = {'core': {'$regex': f'^{MONGO_PREFIX}'}}
        assertion.eq(collection.count_documents(query), n)
        doc = collection.find_one({'ligand smiles': f'{MONGO_PREFIX}CC[O-]'})
    assertion.eq(doc['core'], f'{MONGO_PREFIX}Cd68Cl26Se55')
    assertion.eq(doc['hdf5 index'], {'': 1})


def test_update_mongodb_df(mongo_db: Database) -> None:
    """Test :meth:`~dataCAT.Database.update_mongodb` with a user-provided DataFrame."""
    idx = pd.MultiIndex.from_tuples(
        [(f'{MONGO_PREFIX}CO', 'O2'), (f'{MONGO_PREFIX}CCO', 'O3')], names=['smiles', 'anchor']
    )
    columns = pd.MultiIndex.from_tuples([('E_solv', 'water'), ('hdf5 index', '')])
    df = pd.DataFrame([[-1.5, 0], [-2.5, 1]], index=idx, columns=columns)
    df[('hdf5 index', '')] = df[('hdf5 index', '')].astype(np.int64)

    mongo_db.update_mongodb({'ligand': df})
    docs = _get_mongo_docs(mongo_db, 'ligand_database')
    assertion.eq(docs[f'{MONGO_PREFIX}CCO']['E_solv'], {'water': -2.5})
    assertion.eq(docs[f'{MONGO_PREFIX}CCO']['hdf5 index'], {'': 1})


def test_update_mongodb_raise(mongo_db: Database) -> None:
    """Test :meth:`~dataCAT.Database.update_mongodb` with invalid input."""
    assertion.assert_(mongo_db.update_mongodb, 'bob', exception=ValueError)

    db = Database(DB_PATH_UPDATE)
    db._mongodb = None
    assertion.assert_(db.update_mongodb, 'ligand', exception=ValueError)


@pytest.mark.parametrize('name', ['ligand', 'qd'])
def test_query_mongodb_roundtrip(mongo_db: Database, name: str) -> None:
    """Test that :meth:`~dataCAT.Database.query_mongodb` returns what was exported."""
    mongo_db.update_mongodb(name)
    collection_name, idx_keys = MONGO_COLLECTIONS[name]
    query = {MONGO_KEYS[collection_name]: {'$regex': f'^{MONGO_PREFIX}'}}
    out = mongo_db.query_mongodb(name, query)

    manager = mongo_db.csv_lig if name == 'ligand' else mongo_db.csv_qd
    with manager(write=False) as df:
        ref = df.ndframe.sort_index(axis=1)
    assertion.eq(out.index.names, list(idx_keys))
    assertion.eq(out.columns.names, ref.columns.names)
    pd.testing.assert_frame_equal(out.sort_index(), ref.sort_index())


@pytest.fixture
def mongo_df(mongo_db: Database) -> Database:
    """Export ligands with NaN and missing properties to MongoDB."""
    idx = pd.MultiIndex.from_tuples(
        [(f'{MONGO_PREFIX}{i}', 'O2') for i in ('a', 'b', 'c')], names=['smiles', 'anchor']
    )
    columns = pd.MultiIndex.from_tuples([('cone_angle', 'dist=0.0'), ('V_bulk', '')])
    df = pd.DataFrame([[50.0, 1.0], [150.0, np.nan], [np.nan, 3.0]], index=idx, columns=columns)
    mongo_db.update_mongodb({'ligand': df})

    # A ligand without any cone angle
    idx2 = pd.MultiIndex.from_tuples([(f'{MONGO_PREFIX}d', 'O2')], names=['smiles', 'anchor'])
    df2 = pd.DataFrame([[2.0]], index=idx2, columns=pd.MultiIndex.from_tuples([('V_bulk', '')]))
    mongo_db.update_mongodb({'ligand': df2})
    return mongo_db


QUERY_PARAM = {
    'lt': ({('cone_angle', 'dist=0.0'): {'$lt': 100}}, ['a']),
    'gt': ({('cone_angle', 'dist=0.0'): {'$gt': 100}}, ['b']),
    'empty_sub_index': ({('V_bulk', ''): {'$gte': 2}}, ['c', 'd']),
    'exists': ({('cone_angle', 'dist=0.0'): {'$exists': False}}, ['d']),
    'or': ({'$or': [{('V_bulk', ''): 1.0}, {('cone_angle', 'dist=0.0'): 150.0}]}, ['a', 'b']),
    'index': ({'smiles': f'{MONGO_PREFIX}c'}, ['c']),
}


@pytest.mark.parametrize('query,ref', QUERY_PARAM.values(), ids=QUERY_PARAM.keys())
def test_query_mongodb_filter(mongo_df: Database, query: dict, ref: list) -> None:
    """Test :meth:`~dataCAT.Database.query_mongodb` with property filters."""
    query = {'$and': [{'smiles': {'$regex': f'^{MONGO_PREFIX}'}}, query]}
    out = mongo_df.query_mongodb('ligand', query)
    smiles = [i[len(MONGO_PREFIX):] for i in out.index.get_level_values('smiles')]
    assertion.eq(sorted(smiles), ref)
    assertion.not_(any(i.startswith('_query') for i, _ in out.columns))


def test_query_mongodb_columns(mongo_df: Database) -> None:
    """Test :meth:`~dataCAT.Database.query_mongodb` with **columns** and **limit**."""
    query = {'smiles': {'$regex': f'^{MONGO_PREFIX}'}}
    out = mongo_df.query_mongodb('ligand', query, columns=['cone_angle'])
    assertion.eq(out.columns.tolist(), [('cone_angle', 'dist=0.0')])
    assertion.len_eq(out, 4)

    out2 = mongo_df.query_mongodb('ligand', query, columns=[('V_bulk', ''), ('bob', '')])
    assertion.eq(out2.columns.tolist(), [('V_bulk', ''), ('bob', '')])
    np.testing.assert_array_equal(out2['V_bulk', ''].sort_values(), [1.0, 2.0, 3.0, np.nan])
    assertion.is_(out2['bob', ''].isnull().all(), np.True_)

    out3 = mongo_df.query_mongodb('ligand', query, limit=2)
    assertion.len_eq(out3, 2)

    out4 = mongo_df.query_mongodb('ligand', {'smiles': f'{MONGO_PREFIX}bob'})
    assertion.len_eq(out4, 0)
    assertion.eq(out4.index.names, ['smiles', 'anchor'])


def test_query_mongodb_placeholder(mongo_db: Database) -> None:
    """Test that :meth:`~dataCAT.Database.query_mongodb` skips the placeholder row."""
    idx = pd.MultiIndex.from_tuples([('-', '-')], names=['smiles', 'anchor'])
    df = pd.DataFrame([[1.0]], index=idx, columns=pd.MultiIndex.from_tuples([('V_bulk', '')]))
    with MongoClient(**mongo_db.mongodb) as client:
        collection = client.cat_database.ligand_database
        existed = collection.count_documents({'smiles': '-', 'anchor': '-'}) > 0
    try:
        mongo_db.update_mongodb({'ligand': df})
        assertion.len_eq(mongo_db.query_mongodb('ligand', {'smiles': '-'}), 0)
    finally:
        if not existed:
            with MongoClient(**mongo_db.mongodb) as client:
                client.cat_database.ligand_database.delete_one({'smiles': '-', 'anchor': '-'})


def test_query_mongodb_raise(mongo_db: Database) -> None:
    """Test :meth:`~dataCAT.Database.query_mongodb` with invalid input."""
    assertion.assert_(mongo_db.query_mongodb, 'bob', exception=ValueError)

    db = Database(DB_PATH_UPDATE)
    db._mongodb = None
    assertion.assert_(db.query_mongodb, 'ligand', exception=ValueError)


def test_mongo_dict_to_df() -> None:
    """Test :func:`dataCAT.functions.mongo_dict_to_df`."""
    with DB.csv_lig(write=False) as df:
        ref = df.ndframe
    docs = df_to_mongo_dict(ref)
    out = mongo_dict_to_df(docs, ref.index.names)
    pd.testing.assert_frame_equal(out, ref)

    out2 = mongo_dict_to_df([], ['smiles', 'anchor'])
    assertion.len_eq(out2, 0)
    assertion.eq(out2.index.names, ['smiles', 'anchor'])
    assertion.eq(out2.columns.names, ['index', 'sub index'])
