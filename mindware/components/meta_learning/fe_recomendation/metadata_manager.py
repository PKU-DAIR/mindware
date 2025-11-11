import os
import pickle
import numpy as np
from collections import Counter
from mindware.datasets.utils import calculate_metafeatures
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS


def get_feature_vector(dataset, task_type=None):
    meta_dir = os.path.dirname(__file__)
    meta_dir = os.path.join(meta_dir, '..')
    meta_dir = os.path.join(meta_dir, 'meta_resource')
    meta_dataset_dir = os.path.join(meta_dir, 'meta_dataset_vec')
    if task_type in CLS_TASKS:
        task_prefix = 'cls'
    elif task_type in RGS_TASKS:
        task_prefix = 'rgs'
    else:
        raise ValueError('Invalid task type %s!' % task_type)
    save_path1 = os.path.join(meta_dataset_dir, '%s_meta_dataset_embedding.pkl' % task_prefix)

    assert os.path.exists(save_path1)
    with open(save_path1, 'rb') as f:
        data1 = pickle.load(f)

    task_id = 'init_%s' % dataset

    if task_id in data1['task_ids']:
        idx = data1['task_ids'].index(task_id)
        return data1['dataset_embedding'][idx]
    else:
        return None


class MetaDataManager(object):
    def __init__(self, metadata_dir, builtin_algorithms, builtin_datasets, metric, resource_n,
                 task_type=None, rep=3):
        self.task_type = task_type
        if task_type in CLS_TASKS:
            self.task_prefix = 'cls'
        elif task_type in RGS_TASKS:
            self.task_prefix = 'rgs'
        else:
            raise ValueError('Invalid task type %s!' % self.task_type)
        self.rep_num = rep
        self.metadata_dir = metadata_dir
        self.builtin_algorithms = builtin_algorithms
        self.builtin_datasets = builtin_datasets
        self.metric = metric
        self.resource_n = resource_n

        self._dataset_embedding = list()
        self._dataset_similarity = list()

        save_path3 = os.path.join(self.metadata_dir, 'meta_dataset_vec', '%s_meta_dataset_preprocessor.pkl' % self.task_prefix)
        with open(save_path3, 'rb') as f:
            data3 = pickle.load(f)

        sel_idx1 = [data3['task_ids'].index(t) for t in self.builtin_datasets]
        sel_idx2 = [data3['algorithms_included'].index(t) for t in self.builtin_algorithms]
        self._best_preprocessor = {}
        self._def_preprocessor = {}
        self._sup_preprocessor = {}

        best_preprocessor = data3['best_preprocessor'][self.metric]
        default_preprocesor = data3['default_preprocesor'][self.metric]
        from mindware.components.config_space.cs_builder import get_fe_cs
        full_pres = list(get_fe_cs(self.task_type, silence=True)['preprocessor'].choices)
        for idx2 in sel_idx2:
            self._best_preprocessor[data3['algorithms_included'][idx2]] = [best_preprocessor[idx1][idx2] for idx1 in sel_idx1]
            self._def_preprocessor[data3['algorithms_included'][idx2]] = [default_preprocesor[idx1][idx2] for idx1 in sel_idx1]

            counts = Counter([t for tmp in self._def_preprocessor[data3['algorithms_included'][idx2]] for t in tmp]).most_common()
            tmp = [t[0] for t in counts if t[0] != 'empty']
            for t in full_pres:
                if t != 'empty' and t not in tmp:
                    tmp.append(t)
            self._sup_preprocessor[data3['algorithms_included'][idx2]] = tmp

    def fetch_meta_runs(self, dataset):
        meta_dataset_dir = os.path.join(self.metadata_dir, 'meta_dataset_vec')
        save_path2 = os.path.join(meta_dataset_dir, '%s_meta_dataset_similarity.pkl' % self.task_prefix)
        assert os.path.exists(save_path2)

        with open(save_path2, 'rb') as f:
            data2 = pickle.load(f)

        task_id = 'init_%s' % dataset
        idx = data2['task_ids'].index(task_id)
        sel_idx1 = [data2['task_ids'].index(t) for t in self.builtin_datasets]
        sel_idx2 = [data2['algorithms_included'].index(t) for t in self.builtin_algorithms]
        return data2['similarity'][self.metric][sel_idx2][:, idx, sel_idx1]
    
    def load_meta_data(self):
        meta_dataset_dir = os.path.join(self.metadata_dir, 'meta_dataset_vec')
        save_path1 = os.path.join(meta_dataset_dir, '%s_meta_dataset_embedding.pkl' % self.task_prefix)
        save_path2 = os.path.join(meta_dataset_dir, '%s_meta_dataset_similarity.pkl' % self.task_prefix)

        with open(save_path1, 'rb') as f:
            data1 = pickle.load(f)
        with open(save_path2, 'rb') as f:
            data2 = pickle.load(f)
        _X = list()
        for task_id in self.builtin_datasets:
            idx = data1['task_ids'].index(task_id)
            _X.append(data1['dataset_embedding'][idx])

        self._dataset_embedding = np.asarray(_X)
        _dataset_similarity = data2['similarity'][self.metric]
        sel_idx1 = [data2['task_ids'].index(t) for t in self.builtin_datasets]
        sel_idx2 = [data2['algorithms_included'].index(t) for t in self.builtin_algorithms]
        self._dataset_similarity = _dataset_similarity[sel_idx2][:, sel_idx1][:, :, sel_idx1]

        return self._dataset_embedding, self._dataset_similarity

    def add_meta_runs(self, task_id, dataset_vec, algo_perf):
        pass

    def update2file(self):
        pass
