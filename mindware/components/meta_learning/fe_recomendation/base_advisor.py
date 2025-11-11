import os
import hashlib
import numpy as np
import pickle as pkl
from collections import OrderedDict
from mindware.datasets.utils import calculate_metafeatures
from mindware.utils.logging_utils import get_logger
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS
from mindware.components.meta_learning.fe_recomendation.metadata_manager import MetaDataManager
from mindware.components.meta_learning.fe_recomendation.metadata_manager import get_feature_vector
from collections import Counter
import math

_cls_builtin_algorithms = ['lightgbm', 'random_forest', 'libsvm_svc', 'extra_trees', 'liblinear_svc',
                           'k_nearest_neighbors', 'adaboost', 'lda', 'qda', 'gradient_boosting', 'logistic_regression', 'xgboost']

_rgs_builtin_algorithms = ['lightgbm', 'random_forest', 'libsvm_svr', 'extra_trees', 'liblinear_svr',
                           'k_nearest_neighbors', 'adaboost', 'lasso_regression', 'gradient_boosting', 'ridge_regression', 'xgboost']


class BaseAdvisor(object):
    def __init__(self, 
                 task_type=None,
                 metric='bal_acc',
                 rep=3,
                 total_resource=1200,
                 meta_algorithm='lightgbm',
                 exclude_datasets=None,
                 include_algorithms=None,
                 meta_dir=None):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        self.task_type = task_type
        self.meta_algo = meta_algorithm
        self.rep = rep
        if task_type in CLS_TASKS:
            self.algorithms = _cls_builtin_algorithms
            self.n_algo_candidates = len(_cls_builtin_algorithms)
            if metric not in ['acc', 'f1', 'auc']:
                self.logger.info('Meta information about metric-%s does not exist, use accuracy instead.' % str(metric))
                metric = 'acc'
        elif task_type in RGS_TASKS:
            self.algorithms = _rgs_builtin_algorithms
            self.n_algo_candidates = len(_rgs_builtin_algorithms)
            if metric not in ['mse', 'r2', 'mae']:
                self.logger.info('Meta information about metric-%s does not exist, use accuracy instead.' % str(metric))
                metric = 'mse'
        else:
            raise ValueError('Invalid metric: %s.' % metric)
        
        if include_algorithms is not None:
            for ia in include_algorithms:
                assert ia in self.algorithms

            self.algorithms = include_algorithms

        self.metric = metric

        self.total_resource = total_resource
        self.exclude_datasets = exclude_datasets

        builtin_loc = os.path.dirname(__file__)
        builtin_loc = os.path.join(builtin_loc, '..')
        builtin_loc = os.path.join(builtin_loc, 'meta_resource')
        self.meta_dir = meta_dir if meta_dir is not None else builtin_loc

        if task_type in CLS_TASKS:
            task_prefix = 'cls'
        else:
            task_prefix = 'rgs'

        meta_datasets = set()
        _folder = os.path.join(self.meta_dir, 'meta_dataset_vec')
        embedding_path = os.path.join(_folder, '%s_meta_dataset_embedding.pkl' % task_prefix)
        with open(embedding_path, 'rb')as f:
            d = pkl.load(f)
            meta_datasets = d['task_ids']

        if self.exclude_datasets is not None:
            self.exclude_datasets = [t for t in self.exclude_datasets if 'init_%s'%t in meta_datasets]

        if self.exclude_datasets is None:
            self.hash_id = 'none'
        else:
            self.exclude_datasets = list(set(exclude_datasets))
            exclude_str = ','.join(sorted(self.exclude_datasets))
            md5 = hashlib.md5()
            md5.update(exclude_str.encode('utf-8'))
            self.hash_id = md5.hexdigest()

        self._builtin_datasets = []
        for t in sorted(list(meta_datasets)):
            if self.exclude_datasets is None or t[5:] not in self.exclude_datasets:
                self._builtin_datasets.append(t)
        self.n_dataset_candidates = len(self._builtin_datasets)
        self.metadata_manager = MetaDataManager(self.meta_dir, self.algorithms, self._builtin_datasets,
                                                metric, total_resource, task_type=task_type, rep=rep)
        self.meta_learner = None

    def fetch_dataset_set(self, dataset, datanode=None):
        input_vector = get_feature_vector(dataset, task_type=self.task_type)
        if input_vector is None:
            input_dict = calculate_metafeatures(dataset=datanode, task_type=self.task_type)
            sorted_keys = sorted(input_dict.keys())
            input_vector = [input_dict[key] for key in sorted_keys]
        sims_dict = self.predict(input_vector)
        res_dict = {}
        for algo in self.algorithms:
            idxs = np.argsort(-sims_dict[algo])
            sorted_datasets = [self._builtin_datasets[idx][5:] for idx in idxs]
            sorted_scores = [sims_dict[algo][idx] for idx in idxs]
            res_dict[algo] = OrderedDict(zip(sorted_datasets, sorted_scores))
        return res_dict

    def fetch_preprocessor_set(self, dataset, datanode=None):
        sim_dict = self.fetch_dataset_set(dataset, datanode)

        sim_thr = 0.6
        da_thr = 3  # 至少拿三个数据集
        res_dict = {}
        for algo in sim_dict.keys():

            def_dict = self.metadata_manager._def_preprocessor[algo]  # 比默认配置要好的preprocessor
            best_dict = self.metadata_manager._best_preprocessor[algo]
            _sim_dict = sim_dict[algo]
            res_dict[algo] = []

            dataset_idxs = []
            sims = []
            for key, value in _sim_dict.items():
                if value < sim_thr and len(dataset_idxs) > da_thr:
                    break

                dataset_idx = self._builtin_datasets.index('init_' + key)
                if len(def_dict[dataset_idx]) == 0:
                    continue

                dataset_idxs.append(dataset_idx)
                sims.append(value)

            def_pres = [def_dict[idx] for idx in dataset_idxs]
            best_pres = [best_dict[idx] for idx in dataset_idxs if best_dict[idx] != 'empty']

            # sims = np.exp(sims) / np.sum(np.exp(sims))

            scores = {}
            for i, defs in enumerate(def_pres):
                for _def in defs:
                    if _def not in scores:
                        scores[_def] = 0
                    scores[_def] += sims[i]

            # 先放一个最好的进去
            # for best_pre in best_pres:
            #     if best_pre != 'empty':
            #         res_dict[algo].append(best_pre)
            #         n_preprocessor -= 1
            #         break

            # if len(best_pres)>0:
            #     counts = Counter(best_pres).most_common()
            #     for tmp in counts[:1]:
            #         res_dict[algo].append(tmp[0])
            #         n_preprocessor -= 1

            if len(scores) > 0:
                scores_list = sorted(list(scores.items()), key=lambda x: -x[1])
                for tmp in scores_list:
                    res_dict[algo].append(tmp[0])

            for tmp in self.metadata_manager._sup_preprocessor[algo]:
                if tmp not in res_dict[algo]:
                    res_dict[algo].append(tmp)

        return res_dict

    def fetch_run_results(self, dataset):
        sims_dict = self.metadata_manager.fetch_meta_runs(dataset)
        res_dict = {}
        for i, algo in enumerate(self.algorithms):
            idxs = np.argsort(-sims_dict[i])
            sorted_datasets = [self._builtin_datasets[idx][5:] for idx in idxs]
            sorted_scores = [sims_dict[i][idx] for idx in idxs]
            res_dict[algo] = OrderedDict(zip(sorted_datasets, sorted_scores))
        return res_dict

    def fit(self):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
