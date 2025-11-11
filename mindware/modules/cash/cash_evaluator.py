from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import warnings
import os
import time
import datetime
import numpy as np
import pickle as pkl
from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.evaluators.evaluate_func import holdout_validation, cross_validation, partial_validation
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.constants import *

from mindware.components.evaluators.rgs_evaluator import get_estimator as get_rgs_estimator
from mindware.components.evaluators.cls_evaluator import get_estimator as get_cls_estimator

from mindware.modules.base_evaluator import BaseCLSEvaluator
from mindware.modules.base_evaluator import BaseRGSEvaluator


class CASHCLSEvaluator(BaseCLSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=CLASSIFICATION,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False, reshuffle_ratio=0.0,
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal, reshuffle_ratio
        )


class CASHRGSEvaluator(BaseRGSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=REGRESSION,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1, reshuffle_ratio=0.0,
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed, reshuffle_ratio
        )
