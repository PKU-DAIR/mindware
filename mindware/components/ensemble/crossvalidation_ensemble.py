import pickle as pkl
import numpy as np
from sklearn.metrics._scorer import _BaseScorer, _ThresholdScorer, _PredictScorer
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from mindware.components.metrics.metric import get_metric
from typing import Union, Callable
from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.feature_engineering.parse import construct_node
from mindware.components.utils.constants import CLS_TASKS

class CrossValidationEnsembleModel(BaseEnsembleModel):
    def __init__(self, ensemble_size: int,
                 task_type: int,
                 metric: Union[str, Callable, _BaseScorer],
                 output_dir=None, seed=None,
                 predictions=None):
        self.metric = get_metric(metric)
        self.encoder = OneHotEncoder()
        super().__init__(ensemble_size=ensemble_size, ensemble_method='cross_validation',
                         task_type=task_type, metric=self.metric,
                         output_dir=output_dir, seed=seed,
                         predictions=predictions)
        self.cv_folds = 10
        self.best_model_idx = None
        self.model_weights = []  # 用于存储每个模型的权重

    def _load_models(self):
        self.models = []
        self.model_paths = []
        self.train_labels = None

        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                with open(path, 'rb') as f:
                    op_list, model, _ = pkl.load(f)
                print(f"Loaded model from {path}")
                self.model_paths.append(path)

                _node = self.node.copy_()
                _node = construct_node(_node, op_list)
                X, y = _node.data

                if self.task_type in CLS_TASKS:
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
                else:
                    ss = ShuffleSplit(n_splits=1, test_size=0.33, random_state=42)

                for train_index, val_index in ss.split(X, y):
                    X_valid, y_valid = X[val_index], y[val_index]

                if self.train_labels is None:
                    self.train_labels = y_valid
                else:
                    assert self.train_labels.shape == y_valid.shape

                self.models.append(model)

    def fit(self, stats, datanode):
        super(CrossValidationEnsembleModel, self).fit(stats, datanode)
        self._choose_base_models(datanode)
        if len(self.train_labels.shape) == 1 and self.task_type in CLS_TASKS:
            reshape_y = np.reshape(self.train_labels, (len(self.train_labels), 1))
            self.encoder.fit(reshape_y)

        self._load_models()  # 加载模型
        X, y = datanode.data  # 获取数据和标签

        # 根据任务类型初始化最佳分数
        best_score = float('inf') if self.task_type not in CLS_TASKS else -float('inf')
        best_model_idx = None
        model_scores = []

        # 选择适当的交叉验证方法
        if self.task_type in CLS_TASKS:
            kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # 遍历所有模型，进行交叉验证
        for model_idx, model in enumerate(self.models):
            cv_scores = []
            skip_model = False

            # 对每个模型进行交叉验证
            for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                try:
                    model.fit(X_train, y_train)  # 训练模型

                    # 对分类和回归任务分别进行预测
                    if self.task_type in CLS_TASKS:
                        y_val_pred = model.predict_proba(X_val)  # 分类任务使用 predict_proba
                    else:
                        y_val_pred = model.predict(X_val)  # 回归任务使用 predict

                    # 计算得分并存储
                    score = self.calculate_score(y_val_pred, y_val)
                    cv_scores.append(score)

                except ValueError as e:
                    print(f"Model {model_idx} encountered an error: {e}. Skipping this model.")
                    skip_model = True
                    break

            # 如果模型出错，跳过此模型
            if skip_model:
                continue

            # 计算该模型的平均得分
            mean_score = np.mean(cv_scores)
            model_scores.append(mean_score)
            print(f"Model {model_idx} CV mean score: {mean_score}")

            # 根据任务类型判断是否更新最佳模型
            if (self.task_type in CLS_TASKS and mean_score > best_score) or \
                    (self.task_type not in CLS_TASKS and mean_score < best_score):
                best_score = mean_score
                best_model_idx = model_idx

        # 记录最佳模型的索引
        self.best_model_idx = best_model_idx
        print(f"Selected best model at index: {self.best_model_idx} with score: {best_score}")

        # 根据交叉验证得分计算权重并归一化
        self.model_weights = np.array(model_scores)
        self.model_weights = self.model_weights / np.sum(self.model_weights)  # 归一化权重

    def calculate_score(self, pred, y_true):
        # 分类任务中的处理
        if isinstance(self.metric, _ThresholdScorer):
            if len(y_true.shape) == 1:
                y_true = self.encoder.transform(np.reshape(y_true, (len(y_true), 1))).toarray()
        elif self.task_type in CLS_TASKS and isinstance(self.metric, _PredictScorer):
            pred = np.argmax(pred, axis=-1)  # 将概率转换为类别
        score = self.metric._score_func(y_true, pred) * self.metric._sign
        return score

    def predict(self, data, refit=False):
        if len(self.models) == 0:
            raise ValueError("The model has not been trained yet. Call `fit` before `predict`.")

        X_test = data.data[0]
        total_proba = None

        # 累积每个模型的加权预测
        for model_idx, model in enumerate(self.models):
            if self.task_type in CLS_TASKS:
                proba = model.predict_proba(X_test)  # 分类任务使用概率预测
            else:
                proba = model.predict(X_test)  # 回归任务直接预测

            if total_proba is None:
                total_proba = proba * self.model_weights[model_idx]  # 加权
            else:
                total_proba += proba * self.model_weights[model_idx]  # 累加加权预测

        # 分类任务返回加权后的 **概率分布**（保持与原来一致），回归任务返回加权后的预测值
        if self.task_type in CLS_TASKS:
            return total_proba  # 返回加权的概率分布，而不是类别
        else:
            return total_proba  # 返回回归预测结果

    def get_ens_model_info(self):
        if self.best_model_idx is None:
            raise ValueError("The model has not been trained yet. Call `fit` before `get_ens_model_info`.")

        return {
            'best_model_idx': self.best_model_idx,
            'ensemble_method': self.ensemble_method,
            'ensemble_size': self.ensemble_size,
            'cv_folds': self.cv_folds,
            'model_weights': self.model_weights
        }
