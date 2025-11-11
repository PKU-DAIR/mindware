import os
import pickle as pkl
import joblib
import hashlib
import numpy as np


def check_mode(mode):
    assert mode in ['partial', 'cv', 'full', 'cvpartial']

def load_combined_transformer_estimator(model_dir, config, timestamp, mode='partial', **kwargs):
    model_path = CombinedTopKModelSaver.get_path_by_config(model_dir, config, timestamp, mode=mode, **kwargs)
    op_list, model, _ = CombinedTopKModelSaver._load(model_path)
    return op_list, model


class BaseTopKModelSaver(object):
    def __init__(self, k, model_dir, identifier):
        self.k = k
        self.model_dir = model_dir
        self.identifier = identifier
        self.sorted_list_path = os.path.join(model_dir, '%s_topk_config.pkl' % identifier)
        self.sorted_dict = None

    @staticmethod
    def get_topk_config(config_path):
        if not os.path.exists(config_path):
            return dict()
        with open(config_path, 'rb') as f:
            content = pkl.load(f)
        return content

    def save_topk_config(self):
        with open(self.sorted_list_path, 'wb') as f:
            pkl.dump(self.sorted_dict, f)


class CombinedTopKModelSaver(BaseTopKModelSaver):
    @staticmethod
    def get_configuration_id(config: dict):
        _config = sorted(config.items(), key=lambda x: x[0])
        data_dict = dict(_config)
        data_list = []
        for key, value in sorted(data_dict.items(), key=lambda t: t[0]):
            if isinstance(value, float):
                value = round(value, 5)
            data_list.append('%s-%s' % (key, str(value)))
        data_id = '_'.join(data_list)
        sha = hashlib.sha1(data_id.encode('utf8'))
        return sha.hexdigest()

    @staticmethod
    def get_parse_path(ori_path, mode='partial', **kwargs):
        check_mode(mode)
        prefix = ''
        if mode.startswith('cv'):
            folds = kwargs['folds']
            prefix = "%s%d_" % (mode, folds)
            shuffle = kwargs.get('shuffle', False)
            if shuffle:
                seed = kwargs['seed']
                prefix = "%s%d_s%d_" % (mode, folds, seed)
        elif mode == 'full':
            prefix = "%s_" % mode
        dir_path, filename = os.path.split(ori_path)
        new_filename = prefix + filename
        return os.path.join(dir_path, new_filename)

    @staticmethod
    def get_path_by_config(output_dir, config, identifier, compress=True, mode='partial', **kwargs):
        if compress or config['algorithm'] in ['extra_trees']:
            path = os.path.join(output_dir, '%s_%s.joblib' % (identifier, CombinedTopKModelSaver.get_configuration_id(config)))
        else:
            path = os.path.join(output_dir, '%s_%s.pkl' % (identifier, CombinedTopKModelSaver.get_configuration_id(config)))

        return CombinedTopKModelSaver.get_parse_path(os.path.abspath(path), mode, **kwargs)

    @staticmethod
    def _save(items, save_path: str):
        if save_path.endswith('joblib'):
            with open(save_path, 'wb') as f:
                joblib.dump(items, f, compress=True)
        elif save_path.endswith('pkl'):
            with open(save_path, 'wb') as f:
                pkl.dump(items, f)
        else:
            raise ValueError("Invalid config path: %s", save_path)

    @staticmethod
    def _load(load_path: str):
        if load_path.endswith('joblib'):
            with open(load_path, 'rb') as f:
                return joblib.load(f)
        elif load_path.endswith('pkl'):
            with open(load_path, 'rb') as f:
                return pkl.load(f)
        elif load_path.endswith('npy'):
            return np.load(load_path)
        else:
            raise ValueError("Invalid config path: %s", load_path)

    @staticmethod
    def save_config(items, save_path):
        if not os.path.exists(save_path):
            CombinedTopKModelSaver._save(items=items, save_path=save_path)
        else:
            _, _, perf = CombinedTopKModelSaver._load(load_path=save_path)
            if items[2] > perf:
                CombinedTopKModelSaver._save(items=items, save_path=save_path)

    def add(self, config, perf, estimator_id):
        """
            perf: the larger, the better.
        :param estimator_id:
        :param config:
        :param perf:
        :return:
        """
        _config = config.copy()
        config = dict(sorted(_config.items(), key=lambda x: x[0]))
        model_path_id = self.get_path_by_config(self.model_dir, config, self.identifier)
        model_path_removed = None
        save_flag, delete_flag = False, False
        self.sorted_dict = self.get_topk_config(self.sorted_list_path)
        sorted_list = self.sorted_dict.get(estimator_id, list())

        # Update existed configs
        for sorted_element in sorted_list:
            # if config == sorted_element[0]:
            if config == sorted_element[0] or model_path_id == sorted_element[2]:  # 用path判断是否相同，因为hash的时候取了5位近似（会出现conf不同但hash相同的情况）
                if perf > sorted_element[1]:
                    sorted_list.remove(sorted_element)
                    for idx, item in enumerate(sorted_list):
                        if perf > item[1]:
                            sorted_list.insert(idx, (config, perf, model_path_id))
                            break
                        if idx == len(sorted_list) - 1:
                            sorted_list.append((config, perf, model_path_id))
                            break
                    self.sorted_dict[estimator_id] = sorted_list
                return True, model_path_id, False, model_path_removed

        if len(sorted_list) == 0:
            sorted_list.append((config, perf, model_path_id))
        else:
            # Sorted list is in a descending order.
            for idx, item in enumerate(sorted_list):
                if perf > item[1]:
                    sorted_list.insert(idx, (config, perf, model_path_id))
                    break
                if idx == len(sorted_list) - 1:
                    sorted_list.append((config, perf, model_path_id))
                    break

        if len(sorted_list) > self.k:
            model_path_removed = sorted_list[-1][2]
            delete_flag = True
            sorted_list = sorted_list[:-1]
        if model_path_id in [item[2] for item in sorted_list]:
            save_flag = True

        self.sorted_dict[estimator_id] = sorted_list

        return save_flag, model_path_id, delete_flag, model_path_removed
