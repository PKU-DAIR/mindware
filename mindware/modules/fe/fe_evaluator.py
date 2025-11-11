from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import warnings
import os
import time
import numpy as np
import pickle as pkl
import datetime
from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.constants import *
from mindware.components.evaluators.evaluate_func import validation

from mindware.components.evaluators.cls_evaluator import get_estimator as get_cls_estimator
from mindware.components.evaluators.rgs_evaluator import get_estimator as get_rgs_estimator

from mindware.modules.base_evaluator import BaseCLSEvaluator
from mindware.modules.base_evaluator import BaseRGSEvaluator


class FECLSEvaluator(BaseCLSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal
        )

    @staticmethod
    def _get_parse_data_node(config, train_node, val_node, if_imbal, record=True):
        data_node, op_list = parse_config(train_node, config, record=record, if_imbal=if_imbal)
        _val_node = val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node


class FERGSEvaluator(BaseRGSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed
        )

    @staticmethod
    def _get_parse_data_node(config, train_node, val_node, if_imbal, record=True):
        data_node, op_list = parse_config(train_node, config, record=record, if_imbal=if_imbal)
        _val_node = val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node
