import os
import time
import json
import datetime
import numpy as np
import pickle as pkl
import warnings

from typing import Union, Callable
from sklearn.metrics._scorer import _BaseScorer

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.metrics.metric import get_metric

from mindware.components.optimizers.smac_ens_optimizer import SMACEnsOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.utils.constants import type_dict

from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.utils.logging_utils import setup_logger, get_logger

class BaseEns(object):

    name = 'ens'

    def __init__(self, task_type: str = None, stats=None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'cv', resampling_params=None,
                 optimizer='smac',
                 time_limit=600, amount_of_resource=None, per_run_time_limit=float(np.inf),
                 output_dir=None, seed=1, n_jobs=1, topk=np.inf, 
                 task_id='test', val_nodes:dict=None, judge='val', **cs_args):

        if optimizer not in ['smac']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if evaluation not in ['cv']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.metric_name = 'unknown'
        if isinstance(metric, str):
            self.metric_name = metric
        self.metric = get_metric(metric)
        self.stats = stats
        self.data_node = data_node.copy_()
        self.evaluation = evaluation
        self.resampling_params = resampling_params if resampling_params is not None else {}
        self.resampling_params['folds'] = self.resampling_params.get('folds', 5)
        self.resampling_params['test_size'] = self.resampling_params.get('test_size', 0.33)
        self.seed = seed

        self.optimizer_name = optimizer
        self.per_run_time_limit = per_run_time_limit
        self.time_limit = time_limit
        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource
        self.inner_iter_num_per_iter = 1

        self.timeout_flag = False
        self.early_stop_flag = False
        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.topk = topk

        from mindware.components.config_space.cs_builder import get_ens_cs
        self.cs = get_ens_cs(**cs_args)

        self.incumbent_perf = -float("INF")
        self.incumbent = None
        self.eval_dict = dict()

        if task_type is None:
            task_type = type_of_target(data_node.data[1])
            if task_type in type_dict:
                task_type = type_dict[task_type]
            else:
                raise ValueError("Invalid Task Type: %s!" % task_type)
        self.task_type = task_type

        self.if_imbal = False
        if self.task_type in CLS_TASKS:
            self.if_imbal = is_imbalanced_dataset(self.data_node)

        self.refit_status = 'none'  # none, partial, full

        self.task_id = task_id

        path = 'ENS-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        from mindware.modules.ens.ens_evaluator import EnsEvaluator
        self.evaluator = EnsEvaluator(
            scorer=self.metric, stats=self.stats, 
            data_node=data_node, task_type=self.task_type,
            resampling_strategy=self.evaluation,
            resampling_params=self.resampling_params,
            timestamp=self.timestamp,
            output_dir=self.output_dir,
            seed=self.seed,
            if_imbal=self.if_imbal,
            val_nodes=val_nodes,
            judge=judge,
            n_jobs=self.n_jobs
        )

        self.judge=judge
        self.optimizer = self.build_optimizer()
        self.es = None
        self.es_list = []
        self.predictions = None

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-ENS-%s(%d)' % (optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def build_optimizer(self, name='ens'):

        optimizer = SMACEnsOptimizer(
            evaluator=self.evaluator, config_space=self.cs, name=name, eval_type=self.evaluation,
            time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
            per_run_time_limit=self.per_run_time_limit,
            inner_iter_num_per_iter=self.inner_iter_num_per_iter,timestamp=self.timestamp,
            output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk
        )

        return optimizer

    def iterate(self, trial_num=None):
        if trial_num is None:
            trial_num = self.inner_iter_num_per_iter

        self.optimizer.inner_iter_num_per_iter = trial_num

        self.optimizer.iterate(budget=self.time_limit + self.timestamp - time.time())
        if time.time() - self.timestamp > self.time_limit:
            self.timeout_flag = True
            if self.timeout_flag:
                self.logger.info(f"Time out({self.time_limit}s)!")
        self.early_stop_flag = self.optimizer.early_stopped_flag
        if self.early_stop_flag:
            self.logger.info(f"Early stop!")

        self.incumbent_perf = self.optimizer.incumbent_perf
        self.incumbent = self.optimizer.get_incumbent_config()
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def run(self, refit='full'):

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        # config = self.incumbent.copy()
        # model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.datetime)
        # _, ensemble_builder, learder_board = CombinedTopKModelSaver._load(model_path)
        # self.es = ensemble_builder

        # self.es_list.append(ensemble_builder)

        # best_model_paths = self.evaluator.best_pool.best_model_paths
        # for i in range(len(best_model_paths)-2, -1, -1):
        #     model_path = best_model_paths[i]
        #     if model_path is None:
        #         break
        #     _, ensemble_builder, _ = CombinedTopKModelSaver._load(model_path)
        #     self.es_list.append(ensemble_builder)

        # if refit != 'partial':
        #     self.es.refit(datanode=self.data_node, mode=refit)

        return self.incumbent_perf

    def _predict(self, test_data: DataNode, refit='full'):

        stack_predictions, refit_stack_predictions = self.evaluator.best_pool.predict(test_data, data_node=self.data_node, refit=refit)

        predictions = {'partial': stack_predictions}

        if refit != 'partial':
            predictions[refit] = refit_stack_predictions

        self.predictions = predictions
        with open(os.path.join(self.output_dir, 'predictions.pkl'), 'wb') as f:
            pkl.dump(self.predictions, f)

        topk = len(stack_predictions)
        preds = []
        # if refit != 'partial':
        #     for k in range(topk, 0, -1):
        #         preds.append(np.mean(refit_stack_predictions[-k:], axis=0))
        #     for k in range(topk, 0, -1):
        #         preds.append(np.mean(refit_stack_predictions[-k:] + stack_predictions[-k:], axis=0))

        # for k in range(topk, 0, -1):
        #     preds.append(np.mean(stack_predictions[-k:], axis=0))
        for k in range(topk, 0, -1):
            preds.append(np.mean(refit_stack_predictions[-k:], axis=0))
        for k in range(topk, 0, -1):
            preds.append(refit_stack_predictions[-k])

        return preds

    def predict(self, test_data: DataNode, refit='full'):
        preds = self._predict(test_data, refit=refit)

        results = []
        for pred in preds:
            if self.task_type in CLS_TASKS:
                results.append(np.argmax(pred, axis=-1))
            else:
                results.append(pred)

        return results

    def predict_proba(self, test_data: DataNode, refit='full'):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        preds = self._predict(test_data, refit=refit)

        return preds

    def get_model_info(self, save=False):
        model_info = dict()

        model_info['best_pool'] = self.evaluator.best_pool.get_best_pool_info()
        model_info['best'] = self.incumbent
        # model_info['best_info'] = self.es.get_ens_model_info()
        leader_board = self.evaluator.leader_board
        judge = self.evaluator.judge
        sorted_head = sorted(list(leader_board['train'].keys()), key=lambda x: (-leader_board[judge][x], -leader_board[f'{judge}_2'][x], -leader_board['train'][x], -leader_board['train_2'][x]))
        model_info['leader_board'] = [f"{head}: {', '.join(['%s-%.8e' % (key, leader_board[key][head]) for key in leader_board.keys()])}" for head in sorted_head]
        model_info['comb_count'] = self.evaluator.comb_count

        opt_trajectory = self.optimizer.get_opt_trajectory()
        if opt_trajectory is not None:
            model_info['opt_trajectory'] = opt_trajectory

        if save:
            with open(os.path.join(self.output_dir, 'best_model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=4)

        return model_info

    def get_conf(self, save=False):
        # 获取对象的配置信息
        conf = {
            'name': self.name,
            'judge': self.judge,
            'task_type': self.task_type,
            'task_id': self.task_id,
            'metric': self.metric_name,
            'optimizer': self.optimizer_name,
            'time_limit': self.time_limit,
            'amount_of_resource': self.amount_of_resource,
            'per_run_time_limit': self.per_run_time_limit,
            'seed': self.seed,
            'if_imbal': self.if_imbal,
        }

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf