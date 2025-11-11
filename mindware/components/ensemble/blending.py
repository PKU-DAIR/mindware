import numpy as np
import warnings
import os
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics._scorer import _BaseScorer

from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS
from mindware.modules.base_evaluator import fetch_predict_results
from mindware.components.utils.topk_saver import CombinedTopKModelSaver

class Avging:
    def __init__(self, task_type, n_dim, ensemble_size):

        self.task_type = task_type
        self.n_dim = n_dim
        self.ensemble_size = ensemble_size

    def stack_predict(self, features):

        # features shape: num_data, n_base_model*n_dim
        assert features.shape[1] == self.ensemble_size * self.n_dim

        data_len = features.shape[0]

        features = features.reshape(data_len, -1, self.n_dim)
        pred = np.average(features, axis=1)

        if self.task_type in CLS_TASKS and self.n_dim == 1:
            pred = np.hstack([1-pred, pred])

        if pred.shape[1] == 1:
            pred = pred.reshape(-1)
        return pred


class Besting:
    def __init__(self, task_type, best_idx, ensemble_size):

        self.task_type = task_type
        self.best_idx = best_idx
        self.ensemble_size = ensemble_size

    def stack_predict(self, features):

        # features shape: num_data, n_base_model*n_dim
        assert features.shape[1] % self.ensemble_size == 0
        n_dim = features.shape[1] // self.ensemble_size

        pred = features[:, self.best_idx * n_dim:(self.best_idx + 1) * n_dim]

        if self.task_type in CLS_TASKS and n_dim == 1:
            pred = np.hstack([1-pred, pred])

        if pred.shape[1] == 1:
            pred = pred.reshape(-1)
        return pred

class NonNegativeLinearRegression:
    def fit(self, X, y):
        from scipy.optimize import nnls
        # 使用 nnls 求解非负最小二乘问题
        self.coef_, _ = nnls(X, y)
        return self

    def predict(self, X):
        # 确保输入是二维数组
        return X @ self.coef_

class Blending(BaseEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int, if_imbal: bool,
                 metric: _BaseScorer, resampling_params = None,
                 output_dir=None, seed=None,
                 meta_learner='weighted',
                 predictions=None, base_model_mask=None):
        super().__init__(stats,
                         ensemble_method='blending',
                         ensemble_size=ensemble_size,
                         task_type=task_type, if_imbal=if_imbal,
                         metric=metric, resampling_params=resampling_params,
                         output_dir=output_dir, seed=seed,
                         predictions=predictions)

        self.base_model_mask = base_model_mask

        if meta_learner == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
            except:
                warnings.warn("Lightgbm is not imported! Blending will use linear model instead!")
                meta_learner = 'linear'

        self.meta_method = meta_learner

    @staticmethod
    def build_meta_learner(meta_method, task_type, last_features, final_label=None, **kwargs):

        if isinstance(final_label, pd.Series):
            final_label = final_label.values

        if meta_method in ['weighted', 'equal']:
            ensemble_size = kwargs.get('ensemble_size', None)
            from mindware.components.ensemble.ensemble_selection import EnsembleSelection
            data_len = last_features.shape[0]
            n_dim = last_features.shape[1] // ensemble_size
            last_features = last_features.reshape(data_len, -1, n_dim).transpose(1, 0, 2)
            if task_type in CLS_TASKS and n_dim == 1:
                last_features = np.concatenate([1-last_features, last_features], axis=2)
            meta_learner = EnsembleSelection(stats=None,
                                            ensemble_size=ensemble_size,
                                            task_type=task_type, if_imbal=kwargs['if_imbal'],
                                            metric=kwargs['metric'],
                                            predictions=last_features)
            if meta_method == 'weighted':
                meta_learner.fit(final_label)
            else:
                meta_learner.equal_fit()
        elif meta_method == 'avging':
            ensemble_size = kwargs.get('ensemble_size', None)
            n_dim = last_features.shape[1] // ensemble_size
            meta_learner = Avging(task_type=task_type, n_dim=n_dim, ensemble_size=ensemble_size)
        elif meta_method.startswith('best_'):
            ensemble_size = kwargs.get('ensemble_size', None)
            best_idx = int(meta_method.split('_')[1][3:])
            meta_learner = Besting(task_type=task_type, best_idx=best_idx, ensemble_size=ensemble_size)
        else:
            # We use Xgboost as default meta-learner
            if task_type in CLS_TASKS:
                if meta_method == 'linear':
                    try:
                        from sklearn.linear_model import LogisticRegression
                    except:
                        from sklearn.linear_model.logistic import LogisticRegression
                    meta_learner = LogisticRegression(max_iter=1000)
                elif meta_method == 'gb':
                    try:
                        from sklearn.ensemble import GradientBoostingClassifier
                    except:
                        from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

                    meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                                n_estimators=250)
                elif meta_method == 'lightgbm':
                    from lightgbm import LGBMClassifier
                    meta_learner = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=150, n_jobs=1, verbose=-1)
            else:
                if meta_method == 'linear':
                    # from sklearn.linear_model import LinearRegression
                    # meta_learner = LinearRegression()
                    meta_learner = NonNegativeLinearRegression()
                elif meta_method == 'lightgbm':
                    from lightgbm import LGBMRegressor
                    meta_learner = LGBMRegressor(max_depth=4, learning_rate=0.05, n_estimators=70, n_jobs=1)

            meta_learner.fit(last_features, final_label)

        return meta_learner

    def fit(self, datanode):
        assert self.meta_method in ['weighted', 'linear', 'lightgbm']
        # Train basic models using a part of training data
        base_features= self.get_features(datanode)

        final_labels = {'train': datanode.data[1]}

        self.meta_learner = self.build_meta_learner(self.meta_method, self.task_type, base_features['train'], final_labels['train'],
                                                    ensemble_size=self.ensemble_size, if_imbal=self.if_imbal, metric=self.metric)

        return self

    def get_feature(self, datanode, mode):
        # Predict the labels via blending
        base_features = None
        model_cnt = 0
        suc_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[model_cnt] == 1:
                    _path = CombinedTopKModelSaver.get_parse_path(path, mode=mode, **self.resampling_params)
                    op_list, estimator, _ = CombinedTopKModelSaver._load(_path)
                    pred = fetch_predict_results(self.task_type, op_list, estimator, datanode)
                    if len(pred.shape) == 1:
                        pred = pred.reshape(-1, 1)
                    n_dim = pred.shape[1] if pred.shape[1] > 2 else 1
                    if base_features is None:
                        num_samples = len(datanode.data[0])
                        base_features = np.zeros((num_samples, self.ensemble_size * n_dim))
                    base_features[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, -n_dim:]
                    suc_cnt += 1

                model_cnt += 1

        return base_features

    def predict(self, data, refit='full'):
        last_features = self.get_feature(data, refit)
        # Get predictions from meta-learner
        if self.meta_method in ['weighted', 'avging']:
            final_pred = self.meta_learner.stack_predict(last_features['test'])
        else:
            if self.task_type in CLS_TASKS:
                final_pred = self.meta_learner.predict_proba(last_features['test'])
            else:
                final_pred = self.meta_learner.predict(last_features['test'])

        return final_pred

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, model_path) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    ens_config.append((algo_id, config, model_path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'blending'
        ens_info['meta_learner'] = self.meta_method
        ens_info['config'] = ens_config
        return ens_info

