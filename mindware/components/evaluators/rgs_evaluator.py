from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import time
import warnings
import os
import numpy as np
import pickle as pkl
from sklearn.metrics._scorer import balanced_accuracy_scorer

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.evaluators.evaluate_func import validation
from mindware.components.feature_engineering.task_space import get_task_hyperparameter_space
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.regression import _regressors, _addons
from mindware.components.utils.constants import *
import psutil


def get_estimator(config, estimator_id):
    regressor_type = estimator_id
    config_ = config.copy()
    config_['%s:random_state' % regressor_type] = 1
    hpo_config = dict()
    for key in config_:
        key_name = key.split(':')[0]
        if regressor_type == key_name:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]

    _candidates = get_combined_candidtates(_regressors, _addons)
    estimator = _candidates[regressor_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        # setattr(estimator, 'n_jobs', 1)
        available_cpu = len(psutil.Process().cpu_affinity())
        setattr(estimator, 'n_jobs', available_cpu)
    return regressor_type, estimator
