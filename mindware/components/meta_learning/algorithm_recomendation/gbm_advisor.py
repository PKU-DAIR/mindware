import os
import numpy as np
import lightgbm as lgb
import pickle as pkl
from mindware.utils.logging_utils import get_logger
from mindware.components.meta_learning.algorithm_recomendation.base_advisor import BaseAdvisor


class GBMAdvisor(BaseAdvisor):
    def __init__(self, 
                 task_type=None,
                 metric='acc',
                 exclude_datasets=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(task_type, metric=metric,
                         meta_algorithm='lightgbm', exclude_datasets=exclude_datasets)
        self.model = None

    @staticmethod
    def create_pairwise_data(X, y):
        n_algo = y.shape[1]
        X1, labels = list(), list()
        _instance_num = 0

        for _X, _y in zip(X, y):
            if np.isnan(_X).any():
                continue
            meta_vec = _X
            for i in range(n_algo):
                for j in range(i+1, n_algo):
                    if not np.isfinite(_y[i]) or not np.isfinite(_y[j]):
                        continue
                    if _y[i] == _y[j]:
                        continue

                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x1 = list(meta_vec.copy())
                    meta_x1.extend(vector_i.copy())
                    meta_x1.extend(vector_j.copy())

                    meta_x2 = list(meta_vec.copy())
                    meta_x2.extend(vector_j.copy())
                    meta_x2.extend(vector_i.copy())

                    meta_label1 = 1 if _y[i] > _y[j] else 0
                    meta_label2 = 1 - meta_label1
                    X1.append(meta_x1)
                    labels.append(meta_label1)
                    X1.append(meta_x2)
                    labels.append(meta_label2)
                    _instance_num += 1
        return np.asarray(X1), np.asarray(labels)

    def fit(self, **meta_learner_config):
        meta_learner_dir = os.path.join(self.meta_dir, "meta_learner", "ranknet_model_%s_%s" % (self.meta_algo, self.metric))
        meta_learner_filename = os.path.join(meta_learner_dir, 'ranknet_model_%s_%s_%s.pth' % (self.meta_algo, self.metric, self.hash_id))
        if not os.path.exists(meta_learner_dir):
            os.makedirs(meta_learner_dir)

        if os.path.exists(meta_learner_filename):
            with open(meta_learner_filename, 'rb') as f:
                self.model = pkl.load(f)
        else:
            _X, _y = self.metadata_manager.load_meta_data()
            # print(_X.shape, _y.shape)
            X, y = self.create_pairwise_data(_X, _y)

            # meta_learner_config_filename = self.meta_dir + 'meta_learner_%s_%s_%s_config.pkl' % (
            #     self.meta_algo, self.metric, 'none')
            # if os.path.exists(meta_learner_config_filename):
            #     with open(meta_learner_config_filename, 'rb') as f:
            #         meta_learner_config = pk.load(f)
            # print(meta_learner_config)
            self.model = lgb.LGBMClassifier(**meta_learner_config)
            print(X.shape, y.shape)
            print('Start to fit LGB Model.')
            self.model.fit(X, y)
            print('Fitting LGB Model finished.')
            with open(meta_learner_filename, 'wb') as f:
                pkl.dump(self.model, f)

    def predict(self, meta_feature):
        n_algo = self.n_algo_candidates
        _X = list()
        for i in range(n_algo):
            for j in range(i + 1, n_algo):
                vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                vector_i[i] = 1
                vector_j[j] = 1

                meta_x = list(meta_feature).copy()
                meta_x.extend(vector_i)
                meta_x.extend(vector_j)
                _X.append(meta_x)

        preds = self.model.predict(_X)

        instance_idx = 0
        scores = np.zeros(n_algo)
        for i in range(n_algo):
            for j in range(i + 1, n_algo):
                if preds[instance_idx] == 1:
                    scores[i] += 1
                else:
                    scores[j] += 1
                instance_idx += 1
        return np.array(scores) / np.sum(scores)
