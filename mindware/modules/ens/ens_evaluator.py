from sklearn.metrics._scorer import balanced_accuracy_scorer, _ThresholdScorer, _PredictScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.modules.base_evaluator import BaseEvaluator

from mindware.utils.logging_utils import get_logger
import datetime
from mindware.modules.base_evaluator import BaseCLSEvaluator, BaseRGSEvaluator
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder
import os
import numpy as np
import warnings
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.ens.ens_utils import equal_ens, better_ens
from copy import deepcopy




class BestPool:

    def __init__(self, topk, max_k, output_dir, datetime, judge, logger):
        self.topk = topk
        self.max_k = max_k
        self.output_dir = output_dir
        self.best_model_paths = [None] * self.topk
        self.best_perfs = [-np.inf] * self.topk
        self.best_configs = [None] * self.topk
        self.inner_idxs = [None] * self.topk
        self.datetime = datetime
        self.judge = judge
        self.logger = logger

    def add_config(self, can_config, learder_board, ensemble_builder, inner_idx):
        idx = 0
        while idx < self.topk:
            # 去掉完全一样的config
            if equal_ens(can_config, self.best_configs[idx]):
                return False
            if not better_ens(can_config, self.best_configs[idx]):
                break
            idx += 1
        if idx == 0:
            return False

        remove_path = self.best_model_paths[0]
        if remove_path is not None:
            remove_idx = self.inner_idxs[0]
            if remove_idx == self.max_k - 1:
                os.remove(self.best_model_paths[0])
                self.logger.info(f"Remove old best ens model: {self.best_model_paths[0]}!")

        for i in range(0, idx-1):
            self.best_model_paths[i] = self.best_model_paths[i+1]
            self.best_perfs[i] = self.best_perfs[i+1]
            self.best_configs[i] = self.best_configs[i+1]
            self.inner_idxs[i] = self.inner_idxs[i+1]

        config = {'ensemble_size': can_config[0]['ensemble_size'], 'ratio': can_config[0]['ratio'], 'dropout': can_config[0]['dropout']}
        model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.datetime)
        if not os.path.exists(model_path):
            self.logger.info(f"Save new best ens model: {model_path}!")
            CombinedTopKModelSaver.save_config([None, ensemble_builder, learder_board], model_path)

        self.best_model_paths[idx-1] = model_path
        self.best_perfs[idx-1] = can_config[1][self.judge]
        self.best_configs[idx-1] = can_config
        self.inner_idxs[idx-1] = inner_idx

        return True

    def get_best_pool_info(self):
        info = []
        for i in range(self.topk-1, -1, -1):
            if self.best_configs[i] is None:
                break
            config = self.best_configs[i][0].copy()
            config.update(self.best_configs[i][1])
            config['model_path'] = self.best_model_paths[i]
            info.append(config)

        return info

    def predict(self, test_data: DataNode, data_node: DataNode=None, refit='full'):
        stack_predictions = [None] * self.topk
        refit_stack_predictions = None
        if refit != 'partial':
            refit_stack_predictions = [None] * self.topk
        model_path_dict = {}
        none_num = 0
        for i in range(self.topk):
            model_path = self.best_model_paths[i]
            if model_path is None:
                none_num += 1
                continue
            if model_path not in model_path_dict:
                model_path_dict[model_path] = []
            model_path_dict[model_path].append(i)

        for model_path, idxs in model_path_dict.items():
            _, ensemble_builder, learder_board = CombinedTopKModelSaver._load(model_path)
            inner_idxs = [self.inner_idxs[idx] for idx in idxs]
            for i in range(np.min(inner_idxs)):
                ensemble_builder.model.best_stack.drop_config(i)
            ensemble_builder.model.clear_stack()

            predictions = ensemble_builder.predict(test_data, 'partial')
            for inner_idx, idx in zip(inner_idxs, idxs):
                stack_predictions[idx] = predictions[inner_idx]

            if self.judge == 'train':
                for inner_idx, idx in zip(inner_idxs, idxs):
                    refit_stack_predictions[idx] = predictions[inner_idx]
            else:
                if refit != 'partial':
                    assert data_node is not None
                    ensemble_builder.refit(datanode=data_node, mode=refit)
                    predictions = ensemble_builder.predict(test_data, refit)
                    for inner_idx, idx in zip(inner_idxs, idxs):
                        refit_stack_predictions[idx] = predictions[inner_idx]

        stack_predictions = stack_predictions[none_num:]
        if refit != 'partial':
            refit_stack_predictions = refit_stack_predictions[none_num:]

        return stack_predictions, refit_stack_predictions


class EnsEvaluator(_BaseEvaluator):
    def __init__(
            self, fixed_config=None, scorer=None, stats=None, data_node=None, task_type=0,
            resampling_strategy='cv', resampling_params=None,
            timestamp=None, output_dir=None, seed=1,
            if_imbal=False, val_nodes:dict=None, judge='val', n_jobs=1
    ):

        self.fixed_config = fixed_config
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.if_imbal = if_imbal
        self.task_type = task_type
        self.stats = stats
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = OneHotEncoder()
        if len(self.data_node.data[1].shape) == 1 and self.task_type in CLS_TASKS:
            reshape_y = np.reshape(self.data_node.data[1], (len(self.data_node.data[1]), 1))
            self.onehot_encoder.fit(reshape_y)
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        self.datetime = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        test_size = 0.33
        if self.resampling_params is not None and 'test_size' in self.resampling_params:
            test_size = self.resampling_params['test_size']
        ss = self._get_spliter('holdout', test_size=test_size, random_state=self.seed)

        self.judge = judge
        train_data = self.data_node.copy_(no_data=True)
        val_data = self.data_node.copy_(no_data=True)
        for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
            _x_train, _y_train = BaseEvaluator._get_index_data(self.data_node.data[0], train_index), BaseEvaluator._get_index_data(self.data_node.data[1], train_index)
            _x_val, _y_val = BaseEvaluator._get_index_data(self.data_node.data[0], test_index), BaseEvaluator._get_index_data(self.data_node.data[1], test_index)
            train_data.data = [_x_train, _y_train]
            val_data.data = [_x_val, _y_val]

        # # reshuffle
        # ss = self._get_spliter('holdout', test_size=test_size, random_state=self.seed * 16)
        # train_data = self.data_node.copy_(no_data=True)
        # val_data = self.data_node.copy_(no_data=True)
        # for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
        #     _x_train, _y_train = self.data_node.data[0][train_index], self.data_node.data[1][train_index]
        #     _x_val, _y_val = self.data_node.data[0][test_index], self.data_node.data[1][test_index]
        #     train_data.data = [_x_train, _y_train]
        #     val_data.data = [_x_val, _y_val]

        if val_nodes is None:
            val_nodes = {}
        if self.judge == 'val':
            val_nodes['val'] = val_data.copy_()
        elif self.judge == 'train':
            train_data = self.data_node.copy_()
        else:
            raise ValueError("Wrong value for opt judge: %s" % self.judge)

        self.train_data = train_data
        self.val_nodes = val_nodes

        self.leader_board = dict()
        self.cache = dict()

        self.topk = 20
        self.max_k = 4
        self.best_pool = BestPool(self.topk, self.max_k, self.output_dir, self.datetime, self.judge, self.logger)
        self.comb_count = 0
        self.n_jobs = n_jobs

        self.ensemble_builder = EnsembleBuilder(self.stats, val_data, self.task_type, self.scorer, resampling_params=self.resampling_params,
                                               output_dir=self.output_dir, seed=self.seed, if_imbal=self.if_imbal, thread=self.n_jobs)


    def _get_spliter(self, resampling_strategy, **kwargs):

        if self.task_type in CLS_TASKS:
            ss = BaseCLSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)
        else:
            ss = BaseRGSEvaluator._get_spliter(resampling_strategy=resampling_strategy, **kwargs)

        return ss

    def calculate_score(self, pred, y_true):
        if isinstance(self.scorer, _ThresholdScorer):
            if len(y_true.shape) == 1:
                y_true = self.onehot_encoder.transform(np.reshape(y_true, (len(y_true), 1))).toarray()
        elif self.task_type in CLS_TASKS and isinstance(self.scorer, _PredictScorer):
            pred = np.argmax(pred, axis=-1)
        score = self.scorer._score_func(y_true, pred) * self.scorer._sign
        return score

    def check_cache(self, base_model_mask, dropout):
        base_idx = np.where(base_model_mask)[0]
        dropout = int(len(base_idx) * dropout / 100)
        key = '_'.join([str(x) for x in base_idx]) + f' - d{dropout}'

        if key in self.cache:
            return True, key
        else:
            return False, key

    def __call__(self, config, **kwargs):

        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()

        if self.fixed_config is not None:
            config.update(self.fixed_config)

        # Prepare data node.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            base_model_mask = self.ensemble_builder.build_ensemble(ensemble_method='stacking', ensemble_size=config['ensemble_size'], ratio=config['ratio']/100, dropout=config['dropout']/100,
                                                                stack_layers=config['stack_layers'], meta_learner=config['meta_learner'], max_k=self.max_k, opt=True, judge=self.judge)
            cache, key = self.check_cache(base_model_mask, config['dropout'])

            if cache:
                learder_board = self.cache[key]
            else:
                stacking = self.ensemble_builder.fit(self.train_data, val_nodes=self.val_nodes)
                learder_board = stacking.leader_board
                self.cache[key] = learder_board
                self.comb_count += 1

                # best_perf = np.max(list(learder_board['val'].values()))
                for idx in range(self.max_k):
                    best_config = deepcopy(self.ensemble_builder.model.best_stack.best_configs[idx])
                    if best_config is None: continue
                    best_config[0].update({'ensemble_size': config['ensemble_size'], 'ratio': config['ratio'], 'dropout': config['dropout']})
                    flag = self.best_pool.add_config(best_config, learder_board, self.ensemble_builder, idx)
                    if not flag:
                        self.ensemble_builder.model.best_stack.drop_config(idx)
                        self.ensemble_builder.model.clear_stack()
                # if better_ens(best_config, self.best_config):
                #     self.best_config = best_config.copy()
                #     model_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.datetime)
                #     if self.best_model_path is not None:
                #         os.remove(self.best_model_path)
                #         self.logger.info(f"Remove old best ens model: {self.best_model_path}!")
                #     self.logger.info(f"Save new best ens model: {model_path}!")
                #     CombinedTopKModelSaver.save_config([None, self.ensemble_builder, learder_board], model_path)
                #     self.best_perf = best_config['val']
                #     self.best_model_path = model_path

        size = config['ensemble_size']
        ratio = config['ratio']
        dropout = config['dropout']
        base = f'ens{size}_r{ratio}_d{dropout}'
        for key in learder_board:
            if key not in self.leader_board:
                self.leader_board[key] = {}
            for leader, objective in learder_board[key].items():
                head = f"{base}-{leader}"
                assert head not in self.leader_board[key]
                self.leader_board[key][head] = objective

        # 取反
        learder_board = {key: -value for key, value in learder_board[self.judge].items()}
        return_dict = dict()
        # Turn it into a minimization problem.
        return_dict['leader_board'] = learder_board

        return return_dict