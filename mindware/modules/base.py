import os
import time
import json
import datetime
import numpy as np
import pickle as pkl
import warnings
import shutil

from typing import Union, Callable
from sklearn.metrics._scorer import _BaseScorer

from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.metrics.metric import get_metric

from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer
from mindware.components.optimizers.mab_optimizer import MabOptimizer

from sklearn.utils.multiclass import type_of_target
from mindware.utils.functions import is_imbalanced_dataset
from mindware.components.utils.constants import type_dict

from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
from mindware.components.utils.topk_saver import load_combined_transformer_estimator, CombinedTopKModelSaver, check_mode

from mindware.components.feature_engineering.parse import construct_node
from mindware.components.ensemble import ensemble_list

from mindware.modules.base_evaluator import fetch_predict_estimator, fetch_predict_results
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.feature_engineering.parse import parse_config
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.modules.base_evaluator import BaseEvaluator, get_kfold_name

import warnings
from sklearn.exceptions import ConvergenceWarning
# 忽略特定的警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class BaseAutoML(object):
    name = 'abstract'

    def __init__(self, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test'):

        self.metric_name = 'unknown'
        if isinstance(metric, str):
            self.metric_name = metric
        self.metric = get_metric(metric)
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
        self.inner_iter_num_per_iter = inner_iter_num_per_iter

        self.timeout_flag = False
        self.early_stop_flag = False
        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.topk = topk
        self.rmfiles = rmfiles

        self.optimizer = None
        self.evaluator = None
        self.cs = None

        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.es = None

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

        self.refit_status = 'none'  # none, partial, full, cv

        self.logger = None
        self.task_id = task_id

    @staticmethod
    def _recommand_models(task_type, task_id, data_node, metric, n_algo, include_algorithms=None):

        from mindware.components.meta_learning.algorithm_recomendation.ranknet_advisor_torch_weight import \
            RankNetAdvisor as RankNetAdvisor_W
        alad = RankNetAdvisor_W(task_type=task_type, metric=metric)  # exclude_datasets=[task_id]
        alad.fit()

        model_candidates = alad.fetch_algorithm_set(task_id, datanode=data_node)
        include_models = list()
        for algo in model_candidates:
            if (include_algorithms is None or algo in include_algorithms) and len(include_models) < n_algo:
                include_models.append(algo)

        return include_models

    @staticmethod
    def _recommand_preps(task_type, task_id, data_node, metric, n_prep, include_algorithms, include_preprocessors=None):

        from mindware.components.meta_learning.fe_recomendation.gbm_advisor import GBMAdvisor
        alad_g = GBMAdvisor(task_type=task_type, metric=metric,
                            include_algorithms=include_algorithms)  # exclude_datasets=[task_id],
        alad_g.fit(save_flag=True)

        preprocessors = alad_g.fetch_preprocessor_set(task_id, datanode=data_node)
        include_preps = dict()
        for algo in preprocessors:
            tmp_prep = list()
            for prep in preprocessors[algo]:
                if (include_preprocessors is None or prep in include_preprocessors) and len(tmp_prep) < n_prep:
                    tmp_prep.append(prep)
            tmp_prep = ['empty'] + sorted(tmp_prep)
            include_preps[algo] = tmp_prep

        return include_preps


    @classmethod
    def _refit_config(cls, config, data_node, task_type, if_imbal=False, resampling_params=None, seed=1, mode='full'):
        check_mode(mode)
        algo_id = config['algorithm']

        if mode == 'cv':
            folds = 5
            if resampling_params is not None and 'folds' in resampling_params:
                folds = resampling_params['folds']

            op_list_dict = dict()
            estimator_dict = dict()
            fold = 1
            for train_node, val_node, _, _ in BaseEvaluator._get_cv_data(task_type=task_type, data_node=data_node,
                                                           resampling_params=resampling_params, seed=seed):
                if cls.name in ['fe', 'cashfe']:
                    train_node, op_list = parse_config(train_node, config, record=True, if_imbal=if_imbal)
                else:
                    op_list = {}

                estimator = fetch_predict_estimator(task_type, algo_id, config,
                                                    train_node.data[0], train_node.data[1],X_val=val_node.data[0], y_val = val_node.data[1],
                                                    weight_balance=train_node.enable_balance,
                                                    data_balance=train_node.data_balance,
                                                    feature_map=data_node.feature_map)
                key = get_kfold_name(folds=folds, fold=fold, seed=seed, shuffle=False)
                op_list_dict[key] = op_list
                estimator_dict[key] = estimator
                fold += 1

            return op_list_dict, estimator_dict

        else:
            if mode == 'partial':
                train_node, val_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=data_node,
                                                                   resampling_params=resampling_params, seed=seed)
            else:
                train_node = data_node 
                val_node = data_node

            if cls.name in ['fe', 'cashfe']:
                train_node, op_list = parse_config(train_node, config, record=True, if_imbal=if_imbal)
            else:
                op_list = {}

            estimator = fetch_predict_estimator(task_type, algo_id, config,
                                                train_node.data[0], train_node.data[1],val_node.data[0], val_node.data[1],
                                                weight_balance=train_node.enable_balance,
                                                data_balance=train_node.data_balance,
                                                feature_map=data_node.feature_map)

            return op_list, estimator

    def _get_logger(self, optimizer_name):
        raise NotImplementedError()

    def build_optimizer(self, name='hpo', **kwargs):

        if self.optimizer_name.startswith("block"):
            tree_id = int(self.optimizer_name.split('_')[1])
            from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_execution_tree, \
                get_opt_node_type
            tree = get_opt_execution_tree(tree_id)
      

            if self.evaluation == 'partial':
                sub_optimizer = 'mfse'
            elif self.evaluation == 'partial_bohb':
                sub_optimizer = 'bohb'
            else:
                sub_optimizer = kwargs.get('sub_optimizer', 'smac')

            fe_config_space_dict = kwargs.get('fe_config_space_dict', None)

            optimizer = get_opt_node_type(tree, 0)(
                node_list=tree, node_index=0,
                evaluator=self.evaluator, cash_config_space=self.cs, name=name, eval_type=self.evaluation,
                time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
                per_run_time_limit=self.per_run_time_limit,
                inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
                sub_optimizer=sub_optimizer, fe_config_space_dict=fe_config_space_dict,
                output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk
            )

            return optimizer

        opt_paras = {}
        if self.optimizer_name == 'mab':
            optimizer_class = MabOptimizer
            if self.evaluation == 'partial':
                opt_paras['sub_optimizer'] = 'mfse'
            elif self.evaluation == 'partial_bohb':
                opt_paras['sub_optimizer'] = 'bohb'
            else:
                opt_paras['sub_optimizer'] = kwargs.get('sub_optimizer', 'smac')
            opt_paras['fe_config_space'] = kwargs.get('fe_config_space', None)

        elif self.evaluation == 'partial':
            optimizer_class = MfseOptimizer
        elif self.evaluation == 'partial_bohb':
            optimizer_class = BohbOptimizer
        else:
            # TODO: Support asynchronous BO
            if self.optimizer_name == 'random_search':
                optimizer_class = RandomSearchOptimizer
            elif self.optimizer_name == 'tpe':
                optimizer_class = TPEOptimizer
            elif self.optimizer_name == 'smac':
                optimizer_class = SMACOptimizer
            else:
                raise ValueError("Invalid optimizer %s" % self.optimizer_name)

        optimizer = optimizer_class(
            evaluator=self.evaluator, config_space=self.cs, name=name, eval_type=self.evaluation,
            time_limit=self.time_limit, evaluation_limit=self.amount_of_resource,
            per_run_time_limit=self.per_run_time_limit,
            inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
            output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs, topk=self.topk,
            **opt_paras
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
        self.incumbent = self.optimizer.incumbent_config
        self.eval_dict = self.optimizer.eval_dict
        return self.incumbent_perf

    def rm_files(self):
        self.logger.info('Start to delete files other than incumbent!')
        incumbent_id = CombinedTopKModelSaver.get_configuration_id(self.incumbent)
        for file in os.listdir(self.output_dir):
            if incumbent_id in file or file.endswith('.log') or file.endswith('.json') or file.endswith(
                    'topk_config.pkl'):
                continue
            os.remove(os.path.join(self.output_dir, file))

    def run(self, refit='full'):
        check_mode(refit)

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        if self.ensemble_method is not None:
            # if self.evaluation == 'cv':
            #     self.refit(partial=True)

            self.fit_ensemble(refit)  

        self.refit_incumbent(mode=refit)

        if self.rmfiles:
            self.rm_files()

        stats_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)

        return self.incumbent_perf, stats_path

    def refit_incumbent(self, mode):
        check_mode(mode)

        self.logger.debug('Start to refit the best model!')

        if self.incumbent is None:
            raise AssertionError("The best config is None! Please check if all the evaluations are failed!")

        model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.incumbent, self.datetime, mode=mode, **self.resampling_params)
        if os.path.exists(model_path):
            self.logger.debug("The best model has already been refitted before!")
            return

        config = self.incumbent.copy()
        algo_id = config['algorithm']

        # we need to understand if the mode is "full", but for nn , we need to use 'cv' to refit.
        if 'neural_network' not in algo_id or (algo_id == 'Autogluon_wraper' and config['Autogluon_wraper:model_name'] in ['FASTAI', 'NN_TORCH']):
            par_model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.incumbent, self.datetime, mode='partial', **self.resampling_params)
            shutil.copy(par_model_path, model_path)
        else:
            try:
                op_list, estimator = self._refit_config(self.incumbent, self.data_node, task_type=self.task_type,
                                                        if_imbal=self.if_imbal, resampling_params=self.resampling_params, seed=self.seed, mode=mode)
                CombinedTopKModelSaver._save([op_list, estimator, self.incumbent_perf], model_path)
            except:
                print("Error when refit incumbent config, use origin model!")
                par_model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.incumbent, self.datetime, mode='partial', **self.resampling_params)
                shutil.copy(par_model_path, model_path)
            # if mode == 'full':
            #     mode = 'cv'

            # # And if self.evaluation == mode we don't need to refit anymore.
            # if mode != self.evaluation:
            #     op_list, estimator = self._refit_config(self.incumbent, self.data_node, task_type=self.task_type,
            #                                             if_imbal=self.if_imbal, resampling_params=self.resampling_params, seed=self.seed, mode=mode)
            #     CombinedTopKModelSaver._save([op_list, estimator, self.incumbent_perf], model_path)

    # train with whole data
    def refit(self, mode):
        # if partial, holdout training; else, whole-data training
        check_mode(mode)

        self.logger.debug('Start to refit all the well-performed models!')
        config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)

        if not os.path.exists(config_path):
            warnings.warn("Config path %s not found! Please check if all the evaluations are failed!" % config_path)
            return

        with open(config_path, 'rb') as f:
            stats = pkl.load(f)

        for algo_id in stats.keys():
            if algo_id == 'neural_network':
                continue
            model_to_eval = stats[algo_id]
            for idx, (config, perf, path) in enumerate(model_to_eval):
                path = CombinedTopKModelSaver.get_parse_path(path, mode=mode, **self.resampling_params)
                if os.path.exists(path):
                    self.logger.debug("The model has already been refitted before: %s!" % path)
                    continue
                try:
                    op_list, estimator = self._refit_config(config, data_node=self.data_node, task_type=self.task_type,
                                                            if_imbal=self.if_imbal, resampling_params=self.resampling_params, seed=self.seed, mode=mode)
                    CombinedTopKModelSaver._save([op_list, estimator, perf], path)
                except:
                    self.logger.error("Failed to refit for %s !" % path)

        self.refit_status = mode

    def fit_ensemble(self, refit):
        check_mode(refit)

        self.logger.debug('Start to fit ensemble model!')

        if self.ensemble_method is not None:

            config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.datetime)
            with open(config_path, 'rb') as f:
                stats = pkl.load(f)

            _, valid_node = BaseEvaluator._get_train_valid_data(task_type=self.task_type, data_node=self.data_node,
                                                            resampling_params=self.resampling_params, seed=self.seed)
            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats, valid_node=valid_node,
                                      task_type=self.task_type, if_imbal=self.if_imbal,
                                      metric=self.metric, resampling_params=self.resampling_params,
                                      output_dir=self.output_dir, seed=self.seed)
            self.es.build_ensemble(ensemble_method=self.ensemble_method, ensemble_size=self.ensemble_size)
            datanode = self.data_node if self.ensemble_method == 'stacking' else valid_node
            self.es.fit(datanode=datanode)

            if refit != 'partial':
                self.es.refit(datanode=self.data_node, mode=refit)
        else:
            raise ValueError("No ensemble method is specified!")

    def predict(self, test_data: DataNode, refit='full', ens=True):
        pred = self._predict(test_data, refit=refit, ens=ens)

        if self.task_type in CLS_TASKS:
            return np.argmax(pred, axis=-1)
        else:
            return pred

    @classmethod
    def _fix_model(cls, task_type, data_node: DataNode = None, stats=None, resampling_params=None, seed=1,
                   if_imbal=False):
        metric = get_metric('mse')
        for algo_id in stats.keys():
            if algo_id not in ['xgboost']:
                continue
            model_to_eval = stats[algo_id]
            for idx, (config, perf, path) in enumerate(model_to_eval):
                # _, _, perf = CombinedTopKModelSaver._load(path)
                train_node, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=data_node,
                                                 resampling_params=resampling_params, seed=seed)

                op_list, estimator = cls._refit_config(config, data_node=train_node, task_type=task_type,
                                                       if_imbal=if_imbal)

                valid_node = construct_node(valid_node, op_list)

                pred = estimator.predict(valid_node.data[0])
                new_perf = metric._score_func(pred, valid_node.data[1]) * metric._sign

                CombinedTopKModelSaver._save([op_list, estimator, new_perf], path)

    @classmethod
    def _predict_stats(cls, task_type, metric: Union[str, Callable, _BaseScorer] = None, data_node: DataNode = None,
                       test_data: DataNode = None, stats=None,
                       resampling_params=None,
                       ensemble_method=None, ensemble_size=None, refit='full', prob=False, output_dir='./data', seed=1,
                       task_id='test', thread=2, **ens_args):
        print("Predicting with stats")
        if resampling_params is None:
            resampling_params = {}
        resampling_params['folds'] = resampling_params.get('folds', 3)
        resampling_params['test_size'] = resampling_params.get('test_size', 0.33)

        _path = 'STA-(%d)-%s_%s' % (
            seed, task_id, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        )
        output_dir = os.path.join(output_dir, _path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger_name = 'MindWare-STA-(%d)' % (seed)
        setup_logger(os.path.join(output_dir, '%s.log' % str(logger_name)))
        logger = get_logger(logger_name)

        if_imbal = False
        if task_type in CLS_TASKS:
            if_imbal = is_imbalanced_dataset(data_node)

        # cls._fix_model(task_type=task_type, data_node=data_node, stats=stats, resampling_params=resampling_params, seed=seed, if_imbal=if_imbal)

        stats = stats.copy()
        metric_name = 'unknown'
        if isinstance(metric, str):
            metric_name = metric
        metric = get_metric(metric)

        best_path = None
        best_config = None
        best_perf = -float("INF")
        for algo_id in stats.keys():
            model_to_eval = stats[algo_id]
            for idx, (config, perf, path) in enumerate(model_to_eval):
                if perf > best_perf:
                    best_perf = perf
                    best_config = config
                    best_path = path

        if ensemble_method is None:

            if best_path is None:
                raise AttributeError("No stats found!")

            logger.info('Start to refit the best model! mode: %s!' % refit)
            par_best_path = best_path
            best_path = CombinedTopKModelSaver.get_parse_path(best_path, mode=refit, **resampling_params)
            if os.path.exists(best_path):
                logger.info("The best model has been refitted!")
                best_op_list, estimator, _ = CombinedTopKModelSaver._load(best_path)
            else:
                try:
                    best_op_list, estimator = cls._refit_config(best_config, data_node, task_type=task_type, if_imbal=if_imbal, resampling_params=resampling_params, seed=seed, mode=refit)
                    CombinedTopKModelSaver._save([best_op_list, estimator, best_perf], best_path)
                except:
                    print("Error when refit incumbent config, use origin model!")
                    if os.path.exists(par_best_path):
                        shutil.copy(par_best_path, best_path)
                    best_op_list, estimator, _ = CombinedTopKModelSaver._load(best_path)

            pred = fetch_predict_results(task_type, best_op_list, estimator, test_data)

        else:
            _, valid_node = BaseEvaluator._get_train_valid_data(task_type=task_type, data_node=data_node,
                                                resampling_params=resampling_params, seed=seed)
            es = EnsembleBuilder(stats=stats, valid_node=valid_node,
                                 task_type=task_type, if_imbal=if_imbal,
                                 metric=metric, resampling_params=resampling_params,
                                 output_dir=output_dir, seed=seed, thread=thread)
            
            start = time.time()
            es.build_ensemble(ensemble_method=ensemble_method, ensemble_size=ensemble_size, judge='train', **ens_args)
            datanode = data_node if ensemble_method == 'stacking' else valid_node
            es.fit(datanode=datanode)
            end = time.time()
            if refit != 'partial':
                es.refit(datanode=data_node, mode=refit)
            pred = es.predict(test_data, refit=refit)

        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            conf = {
                'name': cls.name, 'task_type': task_type, 'task_id': task_id,
                'metric': metric_name, 'seed': seed, 'if_imbal': if_imbal,
                'ensemble_method': ensemble_method, 'ensemble_size': ensemble_size
            }
            json.dump(conf, f, indent=4)
        with open(os.path.join(output_dir, 'best_model_info.json'), 'w') as f:
            if ensemble_method is not None:
                ens_args.update({'ensemble_method': ensemble_method, 'ensemble_size': ensemble_size})
                model_info = {'ens_args': ens_args, 'cost': end-start, 'ensemble': es.get_ens_model_info()}
            else:
                model_info = {'best': (best_config['algorithm'], best_config, best_path)}
            json.dump(model_info, f, indent=4)

        if ensemble_method is None:
            shutil.rmtree(output_dir)

        if task_type in CLS_TASKS:
            if prob:
                return pred
            else:
                return np.argmax(pred, axis=-1)
        else:
            return pred

    def _predict(self, test_data: DataNode, refit='full', ens=True):
        if ens and self.ensemble_method is not None:
            if self.es is None and self.evaluation == 'cv':
                raise AttributeError("Please call refit() for cross-validation!")
            elif self.es is None:
                raise AttributeError("AutoML is not fitted!")
            return self.es.predict(test_data, refit=refit)
        else:
            try:
                best_op_list, estimator = load_combined_transformer_estimator(self.output_dir, self.incumbent,
                                                                              self.datetime, mode=refit, **self.resampling_params)
            except Exception as e:
                if self.evaluation == 'cv':
                    raise AttributeError("Please call refit() for cross-validation!")
                else:
                    raise e

            return fetch_predict_results(self.task_type, best_op_list, estimator, test_data)

    def predict_proba(self, test_data: DataNode, refit='full', ens=True):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data, refit=refit, ens=ens)

    def get_model_info(self, save=False):
        model_info = dict()
        if self.es is not None:
            model_info['ensemble'] = self.es.get_ens_model_info()
        path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, self.incumbent, self.datetime)
        model_info['best'] = (self.incumbent['algorithm'], self.incumbent, path)

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
            'task_type': self.task_type,
            'task_id': self.task_id,
            'metric': self.metric_name,
            'optimizer': self.optimizer_name,
            'time_limit': self.time_limit,
            'amount_of_resource': self.amount_of_resource,
            'per_run_time_limit': self.per_run_time_limit,
            'evaluation': self.evaluation,
            'seed': self.seed,
            'if_imbal': self.if_imbal,
            'ensemble_method': self.ensemble_method,
            'ensemble_size': self.ensemble_size
        }
        if hasattr(self, 'cs_args'):
            conf['cs_args'] = self.cs_args
            for key in conf['cs_args']:
                if isinstance(conf['cs_args'][key], np.bool_):
                    conf['cs_args'][key] = bool(conf['cs_args'][key])

        if hasattr(self, 'filter_params'):
            conf['filter_params'] = self.filter_params

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf
