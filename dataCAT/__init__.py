"""A databasing framework for the Compound Attachment Tools package (CAT)."""

# flake8: noqa: F401,E402,N812

from nanoutils import VersionInfo

from .__version__ import __version__

from CAT import version_info as CAT_VERSION
try:
    from nanoCAT import version_info as NANOCAT_VERSION
except ImportError:
    NANOCAT_VERSION = VersionInfo(-1, -1, -1)

DATACAT_VERSION = VersionInfo.from_str(__version__)
version_info = DATACAT_VERSION
del VersionInfo

from .df_proxy import DFProxy

from .property_dset import (
    create_prop_group,
    create_prop_dset,
    update_prop_dset,
    validate_prop_group,
    prop_to_dataframe,
    index_to_pandas,
)

from .hdf5_log import (
    create_hdf5_log,
    update_hdf5_log,
    reset_hdf5_log,
    log_to_dataframe,
)

from .pdb_array import PDBContainer

from .context_managers import (
    OpenLig,
    OpenQD,
)

from .database import Database

from . import (
    functions,
    testing_utils,
    dtype,
    create_database,
    locking,
)

__author__ = 'B. F. van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'CAT_VERSION', 'NANOCAT_VERSION', 'DATACAT_VERSION',

    'functions', 'testing_utils', 'dtype', 'create_database', 'locking',

    'create_hdf5_log', 'update_hdf5_log', 'reset_hdf5_log',

    'create_prop_group', 'create_prop_dset', 'update_prop_dset',
    'validate_prop_group', 'prop_to_dataframe', 'index_to_pandas',

    'DFProxy',

    'PDBContainer',

    'OpenLig', 'OpenQD',

    'Database',
]
