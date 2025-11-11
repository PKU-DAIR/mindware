import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics._scorer import _BaseScorer
from sklearn.preprocessing import OneHotEncoder

from mindware.components.ensemble.blending import Blending, Besting
from mindware.components.utils.constants import CLS_TASKS
from mindware.modules.base_evaluator import fetch_predict_estimator, fetch_predict_results
from mindware.components.feature_engineering.parse import parse_config, construct_node
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base_evaluator import BaseEvaluator, get_kfold_name
from mindware.components.ensemble.parallel_fit import layer_fit
from copy import deepcopy
from mindware.modules.ens.ens_utils import better_ens
from sklearn.metrics._scorer import _BaseScorer, _PredictScorer, _ThresholdScorer
from mindware.components.ensemble.parallel_fit import layer_fit, save_ori_x, rm_ori_x
from mindware.utils.data_manager import DataManager
from mindware.modules.base_evaluator import BaseEvaluator


class BestStack:

    def __init__(self, max_k):

        self.max_k = max_k
        self.best_configs = [None] * self.max_k
        self.best_heads = [None] * self.max_k
        self.meta_learners = [None] * self.max_k

    def add_config(self, can_config, head, meta_learner):

        assert can_config[0]['stack_layers'] + 1 == int(head.split('-')[1][1:])

        idx = 0
        while idx < self.max_k and better_ens(can_config, self.best_configs[idx]):
            idx += 1
        if idx != 0:
            for i in range(0, idx - 1):
                self.best_configs[i] = self.best_configs[i + 1]
                self.best_heads[i] = self.best_heads[i + 1]
                self.meta_learners[i] = self.meta_learners[i + 1]
            self.best_configs[idx - 1] = can_config
            self.best_heads[idx - 1] = head
            self.meta_learners[idx - 1] = meta_learner

    def drop_config(self, idx):
        self.best_configs[idx] = None
        self.best_heads[idx] = None
        self.meta_learners[idx] = None

    @property
    def max_stack_layers(self):

        max_stack_layers = 0
        for i in range(self.max_k):
            if self.best_configs[i] is None: continue

            max_stack_layers = max(max_stack_layers, self.best_configs[i][0]['stack_layers'])

        return max_stack_layers

    # check whether the best configs only contain one config in the last stack layers
    def max_only_best(self):
        max_stack_layers = self.max_stack_layers
        best_idx = None
        count = 0
        for i in range(self.max_k):
            head = self.best_heads[i]
            if head is None: continue

            tmp = head.split('-')
            if int(tmp[-1][1:]) == max_stack_layers + 1:
                count += 1
                if tmp[0].startswith('best'):
                    best_idx = int(tmp[0].split('_')[1][3:])

        only_best = (count == 1) & (best_idx is not None)

        return only_best, best_idx


class Stacking(Blending):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer, resampling_params = None,
                 output_dir=None, seed=None,
                 meta_learner='weighted', stack_layers = 0, thread=20,
                 skip_connect=True, retain=True, dropout=0, max_k=1,
                 predictions=None, base_model_mask=None, opt=False, judge='val'):
        super().__init__(stats=stats,
                ensemble_size=ensemble_size,
                task_type=task_type, if_imbal=if_imbal,
                metric=metric, resampling_params=resampling_params,
                output_dir=output_dir, seed=seed,
                meta_learner=meta_learner, 
                predictions=predictions, base_model_mask=base_model_mask)

        self.opt = opt
        if not self.opt:
            assert max_k == 1
            assert meta_learner in ['weighted', 'linear', 'lightgbm']
            retain = False

        self.ensemble_method = "stacking"
        self.stack_models = None
        self.layer_loss = []
        self.train_cost = []
        self.skip_connect = skip_connect
        self.retain = retain
        self.dropout = dropout
        self.encoder = OneHotEncoder()

        self.stack_layers = stack_layers
        self.thread = thread
        self.leader_board = {'train': {}}
        self.sfolds = 5
        self.folds = self.sfolds
        self.lock = False

        # self.best_config = None
        self.meta_learner = meta_learner

        self.last_features_record = None
        self.final_labels = None
        self.ori_x = None

        self.best_stack = BestStack(max_k)

        self.last_real_loss = None


        assert judge in ['train', 'val']
        self.judge = judge
        self.base_sms = None
        self.base_ops = None

    # get base features from base models in the training phase
    def get_base_features(self, datanode, val_nodes: dict=None, mode='partial'):
        _output_dir = os.path.join(self.output_dir, 'ensemble_tmp')
        model_cnt = 0
        suc_cnt = 0
        ori_xs = {'train': None}

        stack_configs = [[], []]
        ori_config_paths = []
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    stack_configs[0].append(config)
                    ori_config_paths.append(path)
                    if self.skip_connect:
                        if ori_xs['train'] is None: ori_xs = {'train': []}
                        model_path = path # if mode == 'partial' else CombinedTopKModelSaver.get_parse_path(path, 'full')
                        op_list, _, _ = CombinedTopKModelSaver._load(model_path)

                        ori_x = construct_node(datanode.copy_(), op_list).data[0]
                        ori_xs['train'].append(ori_x)
                        if val_nodes is not None:
                            for key in val_nodes.keys():
                                if key not in ori_xs: ori_xs[key] = []
                                ori_x = construct_node(val_nodes[key].copy_(), op_list).data[0]
                                ori_xs[key].append(ori_x)

                    suc_cnt += 1
                model_cnt += 1

        sms, ops, base_features, _, cost = layer_fit(stack_configs=stack_configs, new_node=datanode, ori_xs=None, n_base_model=len(stack_configs[0]),
                                            task_type=self.task_type, if_imbal=self.if_imbal, seed=self.seed, 
                                            layer=0, thread=self.thread, folds=self.folds, output_dir=_output_dir, ori_config_paths=ori_config_paths, logger=self.logger, mode=mode, val_nodes=val_nodes)

        self.base_sms = sms
        self.base_ops = ops

        if not self.lock:
            self.logger.info(f"Cost of Layer0 training with {self.thread} threads: {cost}s")
            self.train_cost.append(cost)

        return base_features, ori_xs


    # get features from base models in the predicting phase
    def get_feature(self, datanode, mode):
        # Predict the labels via blending
        base_features = None
        ori_xs = None
        model_cnt = 0
        suc_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    estimators = self.base_sms[suc_cnt]
                    op_lists = self.base_ops[suc_cnt]
                    for estimator, op_list in zip(estimators, op_lists):
                        pred = fetch_predict_results(self.task_type, op_list, estimator, datanode)
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)
                        n_dim = pred.shape[1] if pred.shape[1] > 2 else 1
                        if base_features is None:
                            num_samples = len(datanode.data[0])
                            base_features = np.zeros((num_samples, self.ensemble_size * n_dim))
                        base_features[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:] / len(estimators)
                    if self.skip_connect:
                        if ori_xs is None: ori_xs = []
                        model_path = path # if mode == 'partial' else CombinedTopKModelSaver.get_parse_path(path, 'full')
                        op_list, _, _ = CombinedTopKModelSaver._load(model_path)
                        ori_x = construct_node(datanode.copy_(), op_list).data[0]
                        ori_xs.append(ori_x)
                    suc_cnt += 1

                model_cnt += 1

        return base_features, ori_xs


    # calculate the scores of stacking features
    def cal_scores(self, last_features, final_labels, n_base_model):
        n_dim = None
        losses = {}
        for key in last_features.keys():
            if n_dim is None:
                n_dim = last_features[key].shape[1] // n_base_model

            loss = []
            second_loss = []

            for i in range(n_base_model):
                if np.isnan(last_features[key][:, i*n_dim:(i+1)*n_dim]).any():
                    loss.append(-np.inf)
                    second_loss.append(-np.inf)
                else:
                    if self.task_type in CLS_TASKS:
                        if n_dim == 1:
                            tmp = last_features[key][:, i].reshape(-1, 1)
                            tmp = np.hstack([1-tmp, tmp])
                        else:
                            tmp = last_features[key][:, i*n_dim:(i+1)*n_dim]

                        _final_label = final_labels[key]
                        if len(_final_label.shape) == 1:
                            _final_label = self.encoder.transform(np.reshape(_final_label, (len(_final_label), 1))).toarray()
                        second_loss.append(-np.sum((tmp - _final_label) ** 2, axis=1).mean())

                        if isinstance(self.metric, _PredictScorer):
                            tmp = np.argmax(tmp, axis=-1)
                    else:
                        tmp = last_features[key][:, i]
                        _final_label = final_labels[key]
                        second_loss.append(-np.mean((tmp - _final_label) ** 2))

                    _final_label = final_labels[key]
                    if isinstance(self.metric, _ThresholdScorer):
                        if len(_final_label.shape) == 1:
                            _final_label = self.encoder.transform(np.reshape(_final_label, (len(_final_label), 1))).toarray()

                    score = -np.inf
                    try:
                        score = self.metric._score_func(_final_label, tmp) * self.metric._sign
                    except:
                        self.logger.info("Error when cal score!")
                    loss.append(score)
            print(key, np.mean(loss), loss)

            losses[key] = loss
            losses[f'{key}_2'] = second_loss

        return losses

    # calculate stacking meta-learner and register leader board
    def register_leader(self, head_output, new_features, last_features, final_labels, layer):

        n_base_model = self.ensemble_size

        # calculate output of validation and test set
        head_outputs = {'train': head_output}
        can_heads = ['weighted', 'linear']
        if not self.opt:
            can_heads = [self.meta_learner] if layer == self.stack_layers + 1 else []

        for config in can_heads:
            head = f"{config}-L{layer}"
            meta_learner = Blending.build_meta_learner(config, self.task_type, last_features['train'], final_labels['train'],
                                ensemble_size=n_base_model, if_imbal=self.if_imbal, metric=self.metric)

            for key in last_features.keys():
                if key == 'train' and head_outputs['train'] is not None: continue
                pred_features = last_features[key]
                if config in ['weighted', 'avging']:
                    pred = meta_learner.stack_predict(pred_features)
                else:
                    if self.task_type in CLS_TASKS:
                        pred = meta_learner.predict_proba(pred_features)
                    else:
                        pred = meta_learner.predict(pred_features)
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                if key not in head_outputs or head_outputs[key] is None:
                    head_outputs[key] = {}
                head_outputs[key][head] = pred

            for key in last_features.keys():
                if head_outputs[key] is not None:
                    perf = self.cal_scores({key: head_outputs[key][head]}, {key: final_labels[key]}, n_base_model=1)
                    for _key in perf:
                        self.leader_board[_key][head] = perf[_key][0]

            perf_dict = {}
            for key in ['train', 'train_2', self.judge, f'{self.judge}_2']:
                if head in self.leader_board[key]:
                    perf_dict.update({key: self.leader_board[key][head]})
            can_config = ({'meta_learner': config, 'stack_layers': layer - 1}, perf_dict)

            self.best_stack.add_config(can_config, head, meta_learner)

        if self.opt and new_features is not None:
            self.last_real_loss = self.cal_scores(new_features, final_labels, self.ensemble_size)
            perfs = [(self.last_real_loss[self.judge][_idx], self.last_real_loss[f'{self.judge}_2'][_idx], self.last_real_loss['train'][_idx], self.last_real_loss['train_2'][_idx]) for _idx in range(self.ensemble_size)]
            _best_idx = max(enumerate(perfs), key=lambda x:x[1])[0]

            head = f"best_idx{_best_idx}-L{layer+1}"

            for key in self.last_real_loss.keys():
                perf = self.last_real_loss[key][_best_idx]
                self.leader_board[key][head] = perf

            perf_dict = {'train': self.last_real_loss['train'][_best_idx], 'train_2': self.last_real_loss['train_2'][_best_idx]}
            perf_dict.update({self.judge: self.last_real_loss[self.judge][_best_idx], f'{self.judge}_2': self.last_real_loss[f'{self.judge}_2'][_best_idx]})
            can_config = ({'meta_learner': 'best', 'stack_layers': layer}, perf_dict)

            meta_learner = Besting(self.task_type, _best_idx, self.ensemble_size)
            self.best_stack.add_config(can_config, head, meta_learner)


    # clear stack models according to best_stack
    def clear_stack(self):
        self.stack_layers = self.best_stack.max_stack_layers
        for key in self.stack_models.keys():
            layer = int(key.split('_')[1])
            if layer > self.stack_layers:
                self.stack_models[key] = None

        max_only_best, best_idx = self.best_stack.max_only_best()
        # if only the best meta-learner in the last layer, drop other models
        if max_only_best:
            for i in range(len(self.stack_models[f'layer_{self.stack_layers}'])):
                if i != best_idx:
                    self.stack_models[f'layer_{self.stack_layers}'][i] = None


    # train or predict stacking meta-learner during refit or predicting phase
    def check_stack(self, layer, last_features, final_labels, train):

        tar_idxs = []
        for i in range(self.best_stack.max_k):
            if self.best_stack.best_heads[i] is None: continue

            head = self.best_stack.best_heads[i]
            if int(head.split('-')[1][1:]) == layer:
                tar_idxs.append(i)

        predictions = None
        if train:
            for idx in tar_idxs:
                meta_method = self.best_stack.best_configs[idx][0]['meta_learner']
                if meta_method.startswith('best'):
                    continue
                meta_learner = Blending.build_meta_learner(meta_method, self.task_type, last_features['train'], final_labels['train'],
                                                           ensemble_size=self.ensemble_size, if_imbal=self.if_imbal, metric=self.metric)
                self.best_stack.meta_learners[idx] = meta_learner
        else:
            predictions = [None] * self.best_stack.max_k
            for idx in tar_idxs:
                meta_method = self.best_stack.best_configs[idx][0]['meta_learner']
                meta_learner = self.best_stack.meta_learners[idx]
                if meta_method in ['weighted', 'avging'] or meta_method.startswith('best'):
                    final_pred = meta_learner.stack_predict(last_features['test'])
                else:
                    if self.task_type in CLS_TASKS:
                        final_pred = meta_learner.predict_proba(last_features['test'])
                    else:
                        final_pred = meta_learner.predict(last_features['test'])

                predictions[idx] = final_pred

        return predictions

    def forward(self, base_features: dict, final_labels: dict=None, train=False, ori_xs: dict=None):

        _output_dir = os.path.join(self.output_dir, 'ensemble_tmp')
        if train:
            assert 'train' in base_features
            assert final_labels is not None
            assert sorted(list(base_features.keys())) == sorted(list(final_labels.keys()))
            for key in base_features.keys():
                if key not in self.leader_board:
                    self.leader_board[key] = {}
                if f'{key}_2' not in self.leader_board:
                    self.leader_board[f'{key}_2'] = {}
            if not self.lock:
                self.layer_loss = [self.cal_scores(base_features, final_labels, self.ensemble_size)]
                self.stack_models = dict()

        stack_configs = [[], ['weighted', 'linear']]
        if not self.opt:
            stack_configs = [[], []]
        model_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, _) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    stack_configs[0].append(config)
                model_cnt += 1
        n_base_model = len(stack_configs[0])
        _key = list(base_features.keys())[0]
        n_dim = base_features[_key].shape[1] // n_base_model
        last_features = deepcopy(base_features)

        stack_predictions = [None] * self.best_stack.max_k

        if train:
            need_last_train = True # whether need to train the last layer meta-learner
            save_ori_x(ori_xs, _output_dir)

            # every layer: train new stacking models and meta-learner of the last layer
            for layer in range(self.stack_layers):

                fail_mask = np.full(n_base_model, False)
                if self.lock:
                    self.check_stack(layer+1, last_features, final_labels, train)
                    sms = self.stack_models[f'layer_{layer+1}']
                    for i in range(len(sms)):
                        if sms[i] is None: fail_mask[i] = True

                new_node = DataManager(last_features['train'], final_labels['train']).get_data_node(last_features['train'], final_labels['train'])
                val_nodes = {}
                for key in last_features:
                    if key == 'train': continue
                    val_nodes[key] = DataManager(last_features[key], final_labels[key]).get_data_node(last_features[key], final_labels[key])

                sms, _, new_features, head_output, cost = layer_fit(stack_configs=stack_configs, new_node=new_node, ori_xs=ori_xs['train'], n_base_model=n_base_model,
                                                                task_type=self.task_type, if_imbal=self.if_imbal, seed=self.seed,
                                                                layer=layer+1, thread=self.thread, folds=self.sfolds, output_dir=_output_dir, logger=self.logger, metric=self.metric,
                                                                skip_mask=fail_mask, val_nodes=val_nodes, dropout=self.dropout)
                self.stack_models[f'layer_{layer+1}'] = sms
                for i in range(len(sms)):
                    if sms[i] is None: fail_mask[i] = True

                if self.lock:
                    # refit meta_learner
                    fail_mask = np.repeat(fail_mask, n_dim)
                    for key in new_features.keys():
                        last_features[key][:, ~fail_mask] = new_features[key][:, ~fail_mask]
                else:
                    self.register_leader(head_output, new_features, last_features, final_labels, layer+1)
                    self.logger.info(f"Cost of Layer{layer+1} training with {self.thread} threads: {cost}s")
                    self.train_cost.append(cost)

                    # retain only the models that can improve the performance
                    if self.retain:
                        bad_mask = ~ (np.array(self.last_real_loss[self.judge]) > np.array(self.layer_loss[-1][self.judge]) + 1e-10)
                        fail_mask |= bad_mask
                        for t in range(n_base_model):
                            if bad_mask[t]: self.stack_models['layer_%d' % (layer+1)][t] = None

                        self.logger.info("Number of models getting improved: %d" % (n_base_model - fail_mask.sum()))

                    rfail_mask = np.repeat(fail_mask, n_dim)
                    for key in new_features.keys():
                        last_features[key][:, ~rfail_mask] = new_features[key][:, ~rfail_mask]

                    self.layer_loss.append(self.cal_scores(last_features, final_labels, self.ensemble_size))

                    if np.all(fail_mask):
                        self.logger.info("None model gets improved! early stop!")
                        need_last_train = False
                        break

            if self.lock:
                self.check_stack(self.stack_layers+1, last_features, final_labels, train)

            else:
                if not self.opt or need_last_train:
                    new_node = DataManager(last_features['train'], final_labels['train']).get_data_node(last_features['train'], final_labels['train'])
                    _, _, _, head_output, cost = layer_fit(stack_configs=[[], stack_configs[1]], new_node=new_node, ori_xs=None, n_base_model=n_base_model,
                                                        task_type=self.task_type, if_imbal=self.if_imbal, seed=self.seed,
                                                        layer=self.stack_layers+1, thread=self.thread, folds=self.sfolds, output_dir=_output_dir, logger=self.logger, metric=self.metric)

                    self.register_leader(head_output, None, last_features, final_labels, self.stack_layers+1)

                self.clear_stack()  # clear unnecessary stack models
                self.lock = True  # lock the stacking structure after first full training, so that refit only needs to train specific models

            rm_ori_x(ori_xs, _output_dir)


        else:

            new_features = {}
            for layer in range(self.stack_layers):
                predictions = self.check_stack(layer + 1, last_features, final_labels, train)
                for idx, pre in enumerate(predictions):
                    if pre is not None:
                        stack_predictions[idx] = pre
                new_node = DataManager(last_features['test'], None).get_data_node(last_features['test'], None)
                sms = self.stack_models['layer_%d' % (layer+1)]
                for key in last_features.keys():
                    new_features[key] = np.zeros_like(last_features[key])
                    for suc_cnt, config in enumerate(stack_configs[0]):
                        estimators = sms[suc_cnt]
                        if estimators is not None:
                            for estimator in estimators:
                                _new_node = new_node
                                if ori_xs is not None:
                                    _new_node = new_node.copy_(no_data=True)
                                    _new_node.data = (BaseEvaluator._concat(ori_xs[key][suc_cnt], last_features[key]), None)
                                    _new_node.reset_feature_map()
                                pred = fetch_predict_results(self.task_type, {}, estimator, _new_node)
                                if len(pred.shape) == 1:
                                    pred = pred.reshape(-1, 1)
                                new_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] += pred[:, -n_dim:] / len(estimators)
                        else:
                            new_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = last_features[key][:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim]
                last_features = deepcopy(new_features)

            predictions = self.check_stack(self.stack_layers+1, last_features, final_labels, train)
            for idx, pre in enumerate(predictions):
                if pre is not None:
                    stack_predictions[idx] = pre


        return stack_predictions

    def get_ens_model_info(self):
        ens_info = super().get_ens_model_info()
        ens_info['ensemble_method'] = 'stacking'
        ens_info['folds'] = [self.folds, self.sfolds]
        ens_info['stask_layers'] = self.stack_layers
        ens_info['dropout'] = self.dropout
        sorted_head = sorted(list(self.leader_board[self.judge].keys()), key=lambda x: (-self.leader_board[self.judge][x], -self.leader_board[f'{self.judge}_2'][x], -self.leader_board['train'][x], -self.leader_board['train_2'][x]))
        ens_info['leader_board'] = [f"{head}: {', '.join(['%s-%.8e' % (key, self.leader_board[key][head]) for key in self.leader_board.keys()])}" for head in sorted_head]
        if self.ensemble_method == 'weighted':
            ens_info['meta_weighted'] = ','.join(['%.3f' % tmp for tmp in self.meta_learner.weights_])
        ens_info['thread'] = self.thread
        ens_info['train_cost'] = self.train_cost
        ens_info['layer_loss'] = self.layer_loss
        return ens_info

    def fit(self, datanode, val_nodes: dict=None):
        if len(datanode.data[1].shape) == 1 and self.task_type in CLS_TASKS:
            reshape_y = np.reshape(datanode.data[1], (len(datanode.data[1]), 1))
            self.encoder.fit(reshape_y)
        # Train basic models using a part of training data
        _mode = 'partial' if self.judge == 'val' else 'full'
        base_features, ori_xs = self.get_base_features(datanode, val_nodes, mode=_mode)

        final_labels = {'train': datanode.data[1]}
        if val_nodes is not None:
            for key in val_nodes.keys():
                final_labels[key] = val_nodes[key].data[1]

        self.forward(base_features, final_labels, train=True, ori_xs=ori_xs)

        return self

    def predict(self, data, refit='full'):
        base_features, ori_xs = self.get_feature(data, refit)
        stack_predictions = self.forward({'test': base_features}, ori_xs={'test': ori_xs})

        if not self.opt:
            stack_predictions = stack_predictions[0]

        return stack_predictions

    def refit(self, datanode, mode):
        if self.opt:
            base_features, ori_xs = self.get_base_features(datanode, mode=mode)

            final_labels = {'train': datanode.data[1]}
            self.forward(base_features, final_labels, train=True, ori_xs=ori_xs)

