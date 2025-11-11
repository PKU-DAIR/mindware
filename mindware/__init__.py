
from mindware.modules.hpo.base_hpo import BaseHPO as HPO
from mindware.modules.fe.base_fe import BaseFE as FE
from mindware.modules.cashfe.base_cashfe import BaseCASHFE as CASHFE
from mindware.modules.cash.base_cash import BaseCASH as CASH
from mindware.modules.ens.base_ens import BaseEns as ENS
from mindware.utils.data_manager import DataManager as DataManager
from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder

from mindware.components.utils.constants import CLASSIFICATION, REGRESSION

from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons as classifiers_addons
from mindware.components.models.regression import _regressors, _addons as regressors_addons

candidates_classifiers = list(get_combined_candidtates(_classifiers, classifiers_addons).keys())
candidates_regressors = list(get_combined_candidtates(_regressors, regressors_addons).keys())

if 'neural_network' in candidates_classifiers:
    candidates_classifiers.remove('neural_network')
if 'neural_network' in candidates_regressors:
    candidates_regressors.remove('neural_network')

__all__ = [
    "HPO",
    "FE",
    "CASH",
    "CASHFE",
    "ENS",
    "DataManager",
    "EnsembleBuilder",
    "CLASSIFICATION",
    "REGRESSION",
    "candidates_classifiers",
    "candidates_regressors"
]
