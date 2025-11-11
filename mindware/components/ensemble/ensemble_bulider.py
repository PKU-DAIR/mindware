from sklearn.metrics._scorer import _BaseScorer
from mindware.components.ensemble.bagging import Bagging
from mindware.components.ensemble.blending import Blending
from mindware.components.ensemble.crossvalidation_ensemble import CrossValidationEnsembleModel
from mindware.components.ensemble.stacking import Stacking
from mindware.components.ensemble.ensemble_selection import EnsembleSelection
from mindware.components.ensemble.parallel_fit import parallel_predict
from mindware.components.utils.topk_saver import CombinedTopKModelSaver, check_mode
from mindware.components.feature_engineering.parse import construct_node
from mindware.modules.base_evaluator import fetch_predict_results
from mindware.components.utils.constants import CLS_TASKS
import numpy as np
from mindware.utils.logging_utils import get_logger
import pickle as pkl
import concurrent.futures as cfutures
import time
import os

from mindware.components.ensemble.ensemble_utils import choose_base_models_classification, \
    choose_base_models_regression

ensemble_list = ['bagging', 'blending', 'stacking', 'ensemble_selection', 'cross_validation']


class EnsembleBuilder:
    def __init__(self, stats, valid_node,
                 task_type: int, metric: _BaseScorer,
                 resampling_params = None,
                 output_dir=None, seed=None,
                 if_imbal=False,
                 thread=20,
                 ):
        self.model = None
        self.stats = stats
        self.valid_node = valid_node.copy_()
        self.task_type = task_type
        self.metric = metric
        self.resampling_params = resampling_params
        self.output_dir = output_dir
        self.seed = seed

        self.predictions = []
        self.model = None
        self.if_imbal = if_imbal
        self.thread = thread

        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)
        self.predictions = self.build_predictions()

    def build_ensemble(self, ensemble_method, ensemble_size, **kwargs):
        ratio = kwargs.get('ratio', 0.4)
        dropout = kwargs.get('dropout', 0)
        self.logger.info(f"Ensemble size: {ensemble_size}, ratio: {ratio}, dropout: {dropout}")
        stack_layers = kwargs.get('stack_layers', 5)
        meta_learner = kwargs.get('meta_learner', 'auto')

        base_model_mask = None
        if len(self.predictions) < ensemble_size:  # 全选
            self.logger.info("The number of models is less than the ensemble size. "
                             "Set the ensemble size to the number of models.")
            ensemble_size = len(self.predictions)
            base_model_mask = np.full(len(self.predictions), True)
        else:
            if ensemble_size == -1: #选择每个模型里面最好的
                base_model_mask = np.full(len(self.predictions), False)
                model_cnt = 0
                for algo_id in self.stats.keys():
                    base_model_mask[model_cnt] = True
                    model_cnt += len(self.stats[algo_id])

                ensemble_size = np.sum(base_model_mask)

            else:
                if ensemble_method not in ["ensemble_selection"]:
                    y_valid = self.valid_node.data[1]
                    if self.task_type in CLS_TASKS:
                        base_model_mask = choose_base_models_classification(
                            np.array(self.predictions), np.array(y_valid), ensemble_size, ratio=ratio
                        )
                    else:
                        base_model_mask = choose_base_models_regression(
                            np.array(self.predictions), np.array(y_valid), ensemble_size, ratio=ratio
                        )
                    ensemble_size = sum(base_model_mask)

        if ensemble_method == 'bagging':
            self.model = Bagging(stats=self.stats,
                                 ensemble_size=ensemble_size,
                                 task_type=self.task_type, if_imbal=self.if_imbal,
                                 metric=self.metric, resampling_params=self.resampling_params,
                                 output_dir=self.output_dir, seed=self.seed,
                                 predictions=None, base_model_mask=base_model_mask)
        elif ensemble_method == 'blending':
            self.model = Blending(stats=self.stats,
                                  ensemble_size=ensemble_size,
                                  task_type=self.task_type, if_imbal=self.if_imbal,
                                  metric=self.metric, resampling_params=self.resampling_params,
                                  output_dir=self.output_dir, seed=self.seed,
                                  meta_learner=meta_learner, 
                                  dropout=dropout,
                                  predictions=None, base_model_mask=base_model_mask)
        elif ensemble_method == 'stacking':
            max_k = kwargs.get('max_k', 1)
            self.model = Stacking(stats=self.stats,
                                  ensemble_size=ensemble_size,
                                  task_type=self.task_type, if_imbal=self.if_imbal,
                                  metric=self.metric, resampling_params=self.resampling_params,
                                  output_dir=self.output_dir, seed=self.seed,
                                  stack_layers=stack_layers, meta_learner=meta_learner, thread=self.thread,
                                  dropout=dropout, max_k=max_k, retain=kwargs.get('retain', True),
                                  predictions=None, base_model_mask=base_model_mask,
                                  opt=kwargs.get('opt', False), judge=kwargs.get('judge', 'val'))
        elif ensemble_method == 'ensemble_selection':
            self.model = EnsembleSelection(stats=self.stats,
                                           ensemble_size=ensemble_size,
                                           task_type=self.task_type, if_imbal=self.if_imbal,
                                           metric=self.metric, resampling_params=self.resampling_params,
                                           output_dir=self.output_dir, seed=self.seed,
                                           predictions=self.predictions)
        else:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

        return base_model_mask

    def build_predictions(self):
        start = time.time()

        if self.thread == 1:
            predictions = []
            model_cnt = 0
            for algo_id in self.stats.keys():
                model_to_eval = self.stats[algo_id]
                for idx, (config, _, path) in enumerate(model_to_eval):
                    op_list, model, _ = CombinedTopKModelSaver._load(path)
                    _node = self.valid_node.copy_()
                    _node = construct_node(_node, op_list)
                    # X_valid, y_valid = _node.data

                    # if self.task_type in CLS_TASKS:
                    #     y_valid_pred = model.predict_proba(X_valid)
                    # else:
                    #     y_valid_pred = model.predict(X_valid)

                    try:
                        y_valid_pred = fetch_predict_results(self.task_type, op_list, model, _node)
                        predictions.append(y_valid_pred)
                    except Exception as e:
                        print(e)

                    model_cnt += 1
                    
        else:
            output_dir = os.path.join(self.output_dir, 'ensemble_tmp')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            all_model_cnt = 0
            for algo_id in self.stats.keys():
                all_model_cnt += len(self.stats[algo_id])

            predictions = [None] * all_model_cnt
            node_path = os.path.join(output_dir, 'valid_node.pkl')
            with open(node_path, 'wb') as f:
                pkl.dump(self.valid_node, f)

            with cfutures.ProcessPoolExecutor(max_workers=self.thread) as executor:
                fs_wait = set()

                model_cnt = 0
                for algo_id in self.stats.keys():
                    model_to_eval = self.stats[algo_id]
                    for idx, (config, _, path) in enumerate(model_to_eval):
                        kwargs = {
                            'model_idx': model_cnt,'config_path': path, 'task_type': self.task_type, 'node_path': node_path
                        }

                        if len(fs_wait) < self.thread:
                            fs_wait.add(executor.submit(parallel_predict, **kwargs))
                        else:
                            fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
                            fs_wait.add(executor.submit(parallel_predict, **kwargs))
                            for fs in fs_done:
                                model_idx, pred = fs.result()
                                predictions[model_idx] = pred

                        model_cnt += 1

                while len(fs_wait) > 0:
                    fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
                    for fs in fs_done:
                        model_idx, pred = fs.result()
                        predictions[model_idx] = pred

            os.remove(node_path)

        self.logger.info(f"Build predictions cost: {time.time() - start}s!")

        return predictions

    def fit(self, datanode, **kwargs):
        return self.model.fit(datanode, **kwargs)

    def predict(self, data, refit):
        check_mode(refit)
        return self.model.predict(data, refit)

    def refit(self, datanode, mode):
        check_mode(mode)
        assert mode != 'partial'
        return self.model.refit(datanode, mode=mode)

    def get_ens_model_info(self):
        return self.model.get_ens_model_info()
