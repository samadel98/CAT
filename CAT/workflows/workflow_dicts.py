"""A module for loading :class:`WorkFlow` templates."""

import os
from types import MappingProxyType
from typing import Any, TYPE_CHECKING
from collections.abc import Mapping, MutableMapping

import yaml
import numpy as np

from .key_map import (
    OPT,
    MOL,
    HDF5_INDEX,
    JOB_SETTINGS_QD_OPT,
    JOB_SETTINGS_CRS,
    JOB_SETTINGS_BDE,
    JOB_SETTINGS_ASA,
    JOB_SETTINGS_CDFT,
    ASA_INT,
    ASA_STRAIN,
    ASA_E,
    SETTINGS1,
    SETTINGS2,
    SETTINGS_SOLV1,
    SETTINGS_SOLV2,
    SETTINGS_ASA,
    SETTINGS_BDE1,
    SETTINGS_BDE2,
    SETTINGS_CDFT,
    V_BULK,
    CDFT_MU,
    CDFT_CHI,
    CDFT_ETA,
    CDFT_S,
    CDFT_GAMMA,
    CDFT_OMEGA,
    CDFT_NUCLEOFUGE,
    CDFT_ELECTROFUGE,
    CDFT_W_MINUS,
    CDFT_W_PLUS,
    CDFT_ELECTROPHILICITY,
    CDFT_DELTAF_MINUS,
    CDFT_DELTAF_PLUS,
    CDFT_MU_MINUS,
    CDFT_MU_PLUS,
    JOB_SETTINGS_CDFT_XTB,
    SETTINGS_CDFT_XTB,
    CDFT_XTB_IP,
    CDFT_XTB_EA,
    CDFT_XTB_MU,
    CDFT_XTB_CHI,
    CDFT_XTB_ETA,
    CDFT_XTB_S,
    CDFT_XTB_OMEGA,
    CDFT_XTB_NUCLEOFUGE,
    CDFT_XTB_ELECTROFUGE,
    CDFT_XTB_W_MINUS,
    CDFT_XTB_W_PLUS,
    CDFT_XTB_ELECTROPHILICITY,
    CDFT_XTB_MU_MINUS,
    CDFT_XTB_MU_PLUS,
    JOB_SETTINGS_SOLV_XTB,
    SETTINGS_SOLV_XTB,
    JOB_SETTINGS_CONFORMERS,
    SETTINGS_CONFORMERS,
    CONFORMERS_COUNT,
    CONFORMERS_E_REL,
    CONE_ANGLE,
    BRANCH_DISTANCE,
    BRANCH_SIZE,
)

if TYPE_CHECKING:
    from typing import TypedDict

    class _TemplateMapping(TypedDict):
        description: str
        mol_type: str
        template: Mapping[str, tuple[str, ...]]
        import_columns: Mapping[tuple[str, str], Any]
        export_columns: tuple[tuple[str, str], ...]

else:
    _TemplateMapping = 'CAT.workflows.workflow_dicts._TemplateMapping'

__all__ = ['WORKFLOW_TEMPLATE', 'CDFT_XTB_COLUMNS']


#: The columns of the conceptual DFT descriptors computed with xTB, all in eV (or 1/eV)
CDFT_XTB_COLUMNS = (
    CDFT_XTB_IP, CDFT_XTB_EA, CDFT_XTB_MU, CDFT_XTB_CHI, CDFT_XTB_ETA, CDFT_XTB_S,
    CDFT_XTB_OMEGA, CDFT_XTB_NUCLEOFUGE, CDFT_XTB_ELECTROFUGE, CDFT_XTB_W_MINUS,
    CDFT_XTB_W_PLUS, CDFT_XTB_ELECTROPHILICITY, CDFT_XTB_MU_MINUS, CDFT_XTB_MU_PLUS,
)


def _load_templates() -> dict[str, Any]:
    """Load the templates from ``CAT/data/workflow_dicts/workflow_yaml.yaml``."""
    path = os.path.join(os.path.dirname(__file__), 'workflow_yaml.yaml')
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _recursive_mapping_proxy(dct: MutableMapping) -> MappingProxyType:
    """Recursivelly convert **dct**, and all nested mutable mappings, into :class:`MappingProxyType<types.MappingProxyType>` instances."""  # noqa
    for k, v in dct.items():
        if isinstance(v, dict):
            dct[k] = _recursive_mapping_proxy(v)
    return MappingProxyType(dct)


def finalize_templates():
    """Update the templates using :data:`._TEMPLATE_UPDATE`."""
    base_dict = {
        'asa': {'import_columns': {ASA_INT: np.nan, ASA_STRAIN: np.nan, ASA_E: np.nan},
                'export_columns': (JOB_SETTINGS_ASA, SETTINGS_ASA, ASA_INT, ASA_STRAIN, ASA_E)},
        'ligand_opt': {'import_columns': {HDF5_INDEX: -1, OPT: False},
                       'export_columns': (MOL, HDF5_INDEX, OPT, SETTINGS1, SETTINGS2)},
        'qd_attach': {'import_columns': {HDF5_INDEX: -1, OPT: False},
                      'export_columns': (MOL, HDF5_INDEX)},
        'qd_opt': {'import_columns': {HDF5_INDEX: -1, OPT: False},
                   'export_columns': (MOL, HDF5_INDEX, OPT, JOB_SETTINGS_QD_OPT,
                                      SETTINGS1, SETTINGS2)},
        'crs': {'import_columns': {},
                'export_columns': (JOB_SETTINGS_CRS, SETTINGS_SOLV1, SETTINGS_SOLV2)},
        'bde': {'import_columns': {},
                'export_columns': (JOB_SETTINGS_BDE, SETTINGS_BDE1, SETTINGS_BDE2)},
        'forcefield': {'import_columns': {},
                       'export_columns': ()},
        'bulkiness': {'import_columns': {V_BULK: np.nan},
                      'export_columns': (V_BULK,)},
        'multi_ligand': {'import_columns': {HDF5_INDEX: -1, OPT: False},
                         'export_columns': (HDF5_INDEX, OPT)},
        'cdft': {'import_columns': {CDFT_MU: np.nan, CDFT_CHI: np.nan, CDFT_ETA: np.nan,
                                    CDFT_S: np.nan, CDFT_GAMMA: np.nan, CDFT_OMEGA: np.nan,
                                    CDFT_NUCLEOFUGE: np.nan, CDFT_ELECTROFUGE: np.nan,
                                    CDFT_W_MINUS: np.nan, CDFT_W_PLUS: np.nan,
                                    CDFT_ELECTROPHILICITY: np.nan, CDFT_DELTAF_MINUS: np.nan,
                                    CDFT_DELTAF_PLUS: np.nan, CDFT_MU_MINUS: np.nan,
                                    CDFT_MU_PLUS: np.nan},
                 'export_columns': (JOB_SETTINGS_CDFT, SETTINGS_CDFT, CDFT_MU, CDFT_CHI, CDFT_ETA,
                                    CDFT_S, CDFT_GAMMA, CDFT_OMEGA, CDFT_NUCLEOFUGE,
                                    CDFT_ELECTROFUGE, CDFT_W_MINUS, CDFT_W_PLUS,
                                    CDFT_ELECTROPHILICITY, CDFT_DELTAF_MINUS, CDFT_DELTAF_PLUS,
                                    CDFT_MU_MINUS, CDFT_MU_PLUS)},
        'cdft_xtb': {'import_columns': dict.fromkeys(CDFT_XTB_COLUMNS, np.nan),
                     'export_columns': (JOB_SETTINGS_CDFT_XTB, SETTINGS_CDFT_XTB,
                                        *CDFT_XTB_COLUMNS)},
        'solv_xtb': {'import_columns': {},
                     'export_columns': (JOB_SETTINGS_SOLV_XTB, SETTINGS_SOLV_XTB)},
        'conformers': {'import_columns': {CONFORMERS_COUNT: np.nan, CONFORMERS_E_REL: np.nan},
                       'export_columns': (JOB_SETTINGS_CONFORMERS, SETTINGS_CONFORMERS,
                                          CONFORMERS_COUNT, CONFORMERS_E_REL)},
        'cone_angle': {'import_columns': {CONE_ANGLE: np.nan},
                       'export_columns': (CONE_ANGLE,)},
        'branch_distance': {'import_columns': {BRANCH_DISTANCE: "", BRANCH_SIZE: ""},
                            'export_columns': (BRANCH_DISTANCE, BRANCH_SIZE)},
    }

    templates = _load_templates()
    for k, v1 in templates.items():
        v2 = base_dict[k]
        v1.update(v2)

        # Convert all lists stored under the 'template' key into tuples
        for key, list_ in v1['template'].items():
            v1['template'][key] = tuple(list_)
    return _recursive_mapping_proxy(templates)


#: An immutable mapping with additional values for ``CAT/data/workflow_dicts/workflow_yaml.yaml``.
#: Contains values which are generally not as easily represented in the .yaml format.
WORKFLOW_TEMPLATE: Mapping[str, _TemplateMapping] = finalize_templates()
