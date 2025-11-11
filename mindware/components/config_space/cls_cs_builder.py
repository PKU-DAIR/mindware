from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from mindware.components.feature_engineering.task_space import get_task_hyperparameter_space
from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons
from mindware.components.utils.constants import *


def get_hpo_cs(estimator_id, task_type=CLASSIFICATION, **cs_args):
    _candidates = get_combined_candidtates(_classifiers, _addons)
    if estimator_id in _candidates:
        clf_class = _candidates[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    cs = clf_class.get_hyperparameter_search_space(**cs_args)
    return cs


def get_cash_cs(include_algorithms=None, task_type=CLASSIFICATION, **cs_args):
    _candidates = get_combined_candidtates(_classifiers, _addons)
    _candidates.pop('neural_network', None)
    _candidates = list(_candidates)
    if include_algorithms is not None:
        _candidates = [_can for _can in _candidates if _can in include_algorithms]
        if len(_candidates) == 0:
            raise ValueError("No algorithms included! Please check the spelling of the included algorithms!")
    cs = ConfigurationSpace()
    algo = CategoricalHyperparameter('algorithm', _candidates)
    cs.add_hyperparameter(algo)
    for estimator_id in _candidates:
        estimator_cs = get_hpo_cs(estimator_id, **cs_args)
        parent_hyperparameter = {'parent': algo,
                                 'value': estimator_id}
        cs.add_configuration_space(estimator_id, estimator_cs, parent_hyperparameter=parent_hyperparameter)
    return cs


def get_fe_cs(task_type=CLASSIFICATION, include_image=False,
              include_text=False, include_preprocessors=None, if_imbal=False, **cs_args):
    cs = get_task_hyperparameter_space(task_type=task_type, include_image=include_image, include_text=include_text,
                                       include_preprocessors=include_preprocessors, if_imbal=if_imbal, **cs_args)
    return cs


def get_combined_cs(task_type=CLASSIFICATION, include_image=False,
                    include_text=False, include_preprocessors=None, if_imbal=False):
    cash_cs = get_cash_cs(task_type)
    fe_cs = get_fe_cs(task_type,
                      include_image=include_image, include_text=include_text,
                      include_preprocessors=include_preprocessors, if_imbal=if_imbal)
    for hp in fe_cs.get_hyperparameters():
        cash_cs.add_hyperparameter(hp)
    for cond in fe_cs.get_conditions():
        cash_cs.add_condition(cond)
    for bid in fe_cs.get_forbiddens():
        cash_cs.add_forbidden_clause(bid)
    return cash_cs
