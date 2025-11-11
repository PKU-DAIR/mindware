from sklearn.metrics._scorer import _BaseScorer
import numpy as np
import os
import pickle as pkl

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.feature_engineering.parse import construct_node

from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base_evaluator import fetch_predict_results
from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer, resampling_params = None,
                 output_dir=None, seed=None,
                 predictions=None, base_model_mask=None):
        super().__init__(stats,
                         ensemble_method='bagging',
                         ensemble_size=ensemble_size,
                         task_type=task_type, if_imbal=if_imbal,
                         metric=metric, resampling_params=resampling_params,
                         output_dir=output_dir, seed=seed,
                         predictions=predictions)

        self.base_model_mask = base_model_mask

    def fit(self):
        return self

    def predict(self, data, refit='full'):
        model_pred_list = []
        final_pred = []
        # Get predictions from each model
        model_cnt = 0
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):

                if self.base_model_mask[model_cnt] == 1:
                    path = CombinedTopKModelSaver.get_parse_path(path, mode=refit, **self.resampling_params)
                    op_list, model, _ = CombinedTopKModelSaver._load(path)
                    model_pred_list.append(fetch_predict_results(self.task_type, op_list, model, data))
                model_cnt += 1

        # Calculate the average of predictions
        for i in range(len(data.data[0])):
            sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
            pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
            final_pred.append(pred_average)

        return np.array(final_pred)

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    ens_config.append((algo_id, config, path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'bagging'
        ens_info['config'] = ens_config
        return ens_info


