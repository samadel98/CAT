"""A module for storing all column keys used throughout CAT.

Index
-----
.. currentmodule:: CAT.workflows.key_map
.. autosummary::
    KEY_MAP

API
---
.. autodata:: KEY_MAP
    :annotation: : Mapping[str, Tuple[str, str]]

"""

from types import MappingProxyType
from collections.abc import Mapping

#: A :class:`Mapping<collections.abc.Mapping>` containing all column keys used throughout CAT.
KEY_MAP: Mapping[str, tuple[str, str]] = MappingProxyType({
    'OPT': ('opt', ''),
    'MOL': ('mol', ''),
    'FORMULA': ('formula', ''),
    'HDF5_INDEX': ('hdf5 index', ''),
    'JOB_SETTINGS_QD_OPT': ('job_settings_qd_opt', ''),
    'JOB_SETTINGS_CRS': ('job_settings_crs', ''),
    'JOB_SETTINGS_BDE': ('job_settings_BDE', ''),
    'JOB_SETTINGS_ASA': ('job_settings_ASA', ''),
    'JOB_SETTINGS_CDFT': ('job_settings_cdft', ''),
    'CDFT_MU': ('cdft', 'Electronic chemical potential (mu)'),
    'CDFT_CHI': ('cdft', 'Electronegativity (chi=-mu)'),
    'CDFT_ETA': ('cdft', 'Hardness (eta)'),
    'CDFT_S': ('cdft', 'Softness (S)'),
    'CDFT_GAMMA': ('cdft', 'Hyperhardness (gamma)'),
    'CDFT_OMEGA': ('cdft', 'Electrophilicity index (w=omega)'),
    'CDFT_NUCLEOFUGE': ('cdft', 'Dissocation energy (nucleofuge)'),
    'CDFT_ELECTROFUGE': ('cdft', 'Dissociation energy (electrofuge)'),
    'CDFT_W_MINUS': ('cdft', 'Electrodonating power (w-)'),
    'CDFT_W_PLUS': ('cdft', 'Electroaccepting power(w+)'),
    'CDFT_ELECTROPHILICITY': ('cdft', 'Net Electrophilicity'),
    'CDFT_DELTAF_MINUS': ('cdft', 'Global Dual Descriptor Deltaf-'),
    'CDFT_DELTAF_PLUS': ('cdft', 'Global Dual Descriptor Deltaf+'),
    'CDFT_MU_MINUS': ('cdft', 'Electronic chemical potential (mu-)'),
    'CDFT_MU_PLUS': ('cdft', 'Electronic chemical potential (mu+)'),
    'JOB_SETTINGS_CDFT_XTB': ('job_settings_cdft_xtb', ''),
    'CDFT_XTB_IP': ('cdft_xtb', 'Ionization potential (I)'),
    'CDFT_XTB_EA': ('cdft_xtb', 'Electron affinity (A)'),
    'CDFT_XTB_MU': ('cdft_xtb', 'Electronic chemical potential (mu)'),
    'CDFT_XTB_CHI': ('cdft_xtb', 'Electronegativity (chi=-mu)'),
    'CDFT_XTB_ETA': ('cdft_xtb', 'Hardness (eta)'),
    'CDFT_XTB_S': ('cdft_xtb', 'Softness (S)'),
    'CDFT_XTB_OMEGA': ('cdft_xtb', 'Electrophilicity index (w=omega)'),
    'CDFT_XTB_NUCLEOFUGE': ('cdft_xtb', 'Dissociation energy (nucleofuge)'),
    'CDFT_XTB_ELECTROFUGE': ('cdft_xtb', 'Dissociation energy (electrofuge)'),
    'CDFT_XTB_W_MINUS': ('cdft_xtb', 'Electrodonating power (w-)'),
    'CDFT_XTB_W_PLUS': ('cdft_xtb', 'Electroaccepting power (w+)'),
    'CDFT_XTB_ELECTROPHILICITY': ('cdft_xtb', 'Net Electrophilicity'),
    'CDFT_XTB_MU_MINUS': ('cdft_xtb', 'Electronic chemical potential (mu-)'),
    'CDFT_XTB_MU_PLUS': ('cdft_xtb', 'Electronic chemical potential (mu+)'),
    'JOB_SETTINGS_SOLV_XTB': ('job_settings_solv_xtb', ''),
    'ASA_INT': ('ASA', 'E_int'),
    'ASA_STRAIN': ('ASA', 'E_strain'),
    'ASA_E': ('ASA', 'E'),
    'SETTINGS1': ('settings', '1'),
    'SETTINGS2': ('settings', '2'),
    'SETTINGS_SOLV1': ('settings', 'solv 1'),
    'SETTINGS_SOLV2': ('settings', 'solv 2'),
    'SETTINGS_ASA': ('settings', 'ASA 1'),
    'SETTINGS_BDE1': ('settings', 'BDE 1'),
    'SETTINGS_BDE2': ('settings', 'BDE 2'),
    'SETTINGS_CDFT': ('settings', 'cdft 1'),
    'SETTINGS_CDFT_XTB': ('settings', 'cdft_xtb 1'),
    'SETTINGS_SOLV_XTB': ('settings', 'solv_xtb 1'),
    'V_BULK': ('V_bulk', ''),
    'CONE_ANGLE': ('cone_angle', 'dist=0.0'),
    'BRANCH_DISTANCE': ('branch_distance', ''),
    'BRANCH_SIZE': ('branch_size', ''),
})

globals().update(KEY_MAP)

__all__ = list(KEY_MAP.keys())
