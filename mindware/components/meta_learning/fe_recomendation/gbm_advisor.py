import numpy as np
import os
import lightgbm as lgb
import pickle as pkl
from mindware.utils.logging_utils import get_logger
from mindware.components.meta_learning.fe_recomendation.base_advisor import BaseAdvisor
from catboost import CatBoostRegressor


class GBMAdvisor(BaseAdvisor):
    def __init__(self, 
                 task_type=None,
                 metric='acc',
                 exclude_datasets=None,
                 include_algorithms=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(task_type, metric=metric,
                         meta_algorithm='lightgbm', exclude_datasets=exclude_datasets, include_algorithms=include_algorithms)
        self.model = None
        self.embeddings = None

    @staticmethod
    def create_pairwise_data(X, y):

        assert len(X) == len(y)

        n_dataset = y.shape[1]
        X1, labels = list(), list()
        _instance_num = 0

        for i in range(n_dataset):
            for j in range(i+1, n_dataset):
                if np.isnan(y[i,j]):
                    continue

                X1.append(np.concatenate([X[i], X[j]]))
                labels.append(y[i,j])
                _instance_num += 1
        return np.asarray(X1), np.asarray(labels)

    def fit(self, save_flag=False):
        _X, _y = self.metadata_manager.load_meta_data()
        self.embeddings = _X
        self.model = {}

        for i, algo in enumerate(self.algorithms):
            meta_learner_dir = os.path.join(self.meta_dir, "meta_learner", "sim_model_%s_%s" % (self.meta_algo, self.metric))
            meta_learner_filename = os.path.join(meta_learner_dir, 'sim_model_%s_%s_%s_%s.pkl' % (self.meta_algo, self.metric, algo, self.hash_id))
            if save_flag and not os.path.exists(meta_learner_dir):
                os.makedirs(meta_learner_dir)
            if save_flag and os.path.exists(meta_learner_filename):
                # print("load model...")
                with open(meta_learner_filename, 'rb') as f:
                    self.model[algo] = pkl.load(f)
            else:
                # print(_X.shape, _y.shape)
                X, y = self.create_pairwise_data(_X, _y[i])
                surrogate_cat = CatBoostRegressor()
                surrogate_cat.fit(X, y, silent=True)
                # surrogate_cat = lgb.LGBMRegressor(verbose=0)
                # surrogate_cat.fit(X, y)
                self.model[algo] = surrogate_cat
                if save_flag:
                    with open(meta_learner_filename, 'wb') as f:
                        pkl.dump(surrogate_cat, f)

    def predict(self, meta_feature):
        n_dataset = self.n_dataset_candidates
        _X = list()
        for i in range(n_dataset):

            meta_x = np.array(meta_feature).copy()
            meta_y = self.embeddings[i].copy()
            _X.append(np.concatenate((meta_x, meta_y)))

        sims_dict = {}
        for algo in self.algorithms:
            sims_dict[algo] = self.model[algo].predict(_X)

        return sims_dict
