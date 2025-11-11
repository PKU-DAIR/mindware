from mindware.components.utils.constants import *

from mindware.components.feature_engineering.parse import parse_config, construct_node

from mindware.modules.base_evaluator import BaseCLSEvaluator
from mindware.modules.base_evaluator import BaseRGSEvaluator


class CASHFECLSEvaluator(BaseCLSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=CLASSIFICATION,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False, reshuffle_ratio=0
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed,
            if_imbal, reshuffle_ratio
        )

    @staticmethod
    def _get_parse_data_node(config, train_node, val_node, if_imbal, record=True):
        data_node, op_list = parse_config(train_node, config, record=record, if_imbal=if_imbal)
        _val_node = val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node


class CASHFERGSEvaluator(BaseRGSEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, data_node=None, task_type=REGRESSION,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1, reshuffle_ratio =0,
    ):
        super().__init__(
            fixed_config, scorer, data_node, task_type,
            resampling_strategy, resampling_params,
            timestamp, output_dir, seed, reshuffle_ratio
        )

    @staticmethod
    def _get_parse_data_node(config, train_node, val_node, if_imbal, record=True):
        data_node, op_list = parse_config(train_node, config, record=record, if_imbal=if_imbal)
        _val_node = val_node.copy_()
        _val_node = construct_node(_val_node, op_list)

        return op_list, data_node, _val_node
