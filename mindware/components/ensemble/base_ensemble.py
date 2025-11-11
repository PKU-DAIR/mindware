from sklearn.metrics._scorer import _BaseScorer
import time
import datetime
import os
import shutil

from mindware.components.feature_engineering.parse import parse_config
from mindware.modules.base_evaluator import fetch_predict_estimator
from mindware.utils.logging_utils import get_logger
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base_evaluator import BaseEvaluator, get_kfold_name


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, stats,
                 ensemble_method: str,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer, resampling_params = None,
                 output_dir=None, seed=None,
                 predictions=None):
        self.stats = stats
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.resampling_params = resampling_params
        self.output_dir = output_dir
        self.seed = seed

        self.datetime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        logger_name = 'EnsembleBuilder'
        self.logger = get_logger(logger_name)

        self.if_imbal = if_imbal
        self.predictions = predictions

        self.base_model_mask = None

    def fit(self, datanode):
        raise NotImplementedError

    def predict(self, data, refit='full'):
        raise NotImplementedError

    def get_ens_model_info(self):
        raise NotImplementedError

    # TODO: Refit
    def refit(self, datanode, mode):
        self.logger.debug("Start to refit all models needed by ensemble!")
        # Refit models on whole training data
        model_cnt = 0
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, model_path) in enumerate(model_to_eval):
                # X, y = self.node.data
                if self.base_model_mask[model_cnt] == 1:

                    save_path = CombinedTopKModelSaver.get_parse_path(model_path, mode=mode, **self.resampling_params)
                    if os.path.exists(save_path):
                        self.logger.info("Already Refit model %d[%s], path: %s" % (model_cnt, config['algorithm'], save_path))
                        model_cnt += 1
                        continue

                    self.logger.info("Refit model %d[%s], path: %s" % (model_cnt, config['algorithm'], save_path))
                    par_op_list, par_estimator, perf = CombinedTopKModelSaver._load(model_path)

                    if mode == 'full':
                        if 'neural_network' not in algo_id or (algo_id == 'Autogluon_wraper' and config['Autogluon_wraper:model_name'] in ['FASTAI', 'NN_TORCH']):
                            shutil.copy(model_path, save_path)
                        else:
                            try:
                                if par_op_list == {}:
                                    op_list = {}
                                    _node = datanode.copy_()
                                else:
                                    _node, op_list = parse_config(datanode, config, record=True,
                                                                if_imbal=self.if_imbal)

                                estimator = fetch_predict_estimator(self.task_type, config['algorithm'], config,
                                                                    _node.data[0], _node.data[1],
                                                                    weight_balance=_node.enable_balance,
                                                                    data_balance=_node.data_balance)
                                CombinedTopKModelSaver._save(items=[op_list, estimator, perf], save_path=save_path)
                            except:
                                print("Error when refit, use partial model!")
                                CombinedTopKModelSaver._save(items=[par_op_list, par_estimator, perf], save_path=save_path)
                    else:
                        folds = 3
                        if self.resampling_params is not None and 'folds' in self.resampling_params:
                            folds = self.resampling_params['folds']
                        op_list_dict = dict()
                        estimator_dict = dict()
                        fold = 1
                        for train_node, _, _, _ in BaseEvaluator._get_cv_data(task_type=self.task_type, data_node=datanode,
                                                                    resampling_params=self.resampling_params, seed=self.seed):
                            key = get_kfold_name(folds=folds, fold=fold, seed=self.seed, shuffle=False)
                            try:
                                if par_op_list == {}:
                                    op_list = {}
                                    _node = train_node.copy_()
                                else:
                                    _node, op_list = parse_config(train_node, config, record=True,
                                                                if_imbal=self.if_imbal)

                                estimator = fetch_predict_estimator(self.task_type, config['algorithm'], config,
                                                                    _node.data[0], _node.data[1],
                                                                    weight_balance=_node.enable_balance,
                                                                    data_balance=_node.data_balance)

                                op_list_dict[key] = op_list
                                estimator_dict[key] = estimator
                            except:
                                print("Error when refit, use partial model!")
                                op_list_dict[key] = par_op_list
                                estimator_dict[key] = par_estimator

                            fold += 1

                        CombinedTopKModelSaver._save(items=[op_list_dict, estimator_dict, perf], save_path=save_path)

                model_cnt += 1

    @staticmethod
    def get_hyperparameter_search_space():

        raise NotImplementedError
