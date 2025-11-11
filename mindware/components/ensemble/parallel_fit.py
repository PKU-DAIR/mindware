import os
import time
import numpy as np
import pickle as pkl
import pandas as pd
from mindware.modules.base_evaluator import BaseEvaluator, fetch_predict_results, fetch_predict_estimator, get_kfold_name
from mindware.components.utils.constants import CLS_TASKS, RGS_TASKS
import concurrent.futures as cfutures
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.evaluators.cls_evaluator import get_estimator as get_cls_estimator
from mindware.components.evaluators.rgs_evaluator import get_estimator as get_rgs_estimator
from mindware.components.feature_engineering.parse import parse_config, construct_node
import psutil

def get_node_name(layer, key):

    return f'node_layer{layer}_{key}.pkl'


def get_ori_x_name(model_idx):

    return f'ori_x_conf{model_idx}.pkl'


def get_estimator_name(layer, model_idx, fold):

    return f'est_layer{layer}_conf{model_idx}_fold{fold}.joblib'


def _fit(config, task_type, if_imbal, seed, 
         layer, model_idx, fold, folds,
         train_index, valid_index, 
         data_node, ori_x, ori_config_path, shuffle, **kwargs):
    
    # start = time.time()

    train_node = data_node.copy_(no_data=True)
    valid_node = data_node.copy_(no_data=True)
    if ori_x is not None and not isinstance(config, str):
        train_node.data = (BaseEvaluator._concat(
            BaseEvaluator._get_index_data(ori_x['train'], train_index), BaseEvaluator._get_index_data(data_node.data[0], train_index)
            ), 
            BaseEvaluator._get_index_data(data_node.data[1], train_index)
        )
        train_node.reset_feature_map()
        valid_node.data = (BaseEvaluator._concat(
            BaseEvaluator._get_index_data(ori_x['train'], valid_index), BaseEvaluator._get_index_data(data_node.data[0], valid_index)
            ), 
            BaseEvaluator._get_index_data(data_node.data[1], valid_index)
        )
        valid_node.reset_feature_map()
    else:
        train_node.data = (BaseEvaluator._get_index_data(data_node.data[0], train_index), BaseEvaluator._get_index_data(data_node.data[1], train_index))
        valid_node.data = (BaseEvaluator._get_index_data(data_node.data[0], valid_index), BaseEvaluator._get_index_data(data_node.data[1], valid_index))
    op_list = {}
    estimator = None
    need_save = True
    if isinstance(config, str):
        need_save = False
        from mindware.components.ensemble.blending import Blending
        meta_learner = Blending.build_meta_learner(config, task_type, train_node.data[0], train_node.data[1],
                                                    ensemble_size=kwargs['n_base_model'], if_imbal=if_imbal, metric=kwargs['metric'])
        pred_features = valid_node.data[0]
        if config == 'weighted':
            pred = meta_learner.stack_predict(pred_features)
        else:
            if task_type in CLS_TASKS:
                pred = meta_learner.predict_proba(pred_features)
            else:
                pred = meta_learner.predict(pred_features)

        if len(pred.shape) == 1: pred = pred.reshape(-1, 1)
        preds = {'train': pred}
    else:
        if layer == 0:
            _mode = 'cv'
            if kwargs['mode'] == 'partial':
                _mode += kwargs['mode']
            cv_path = CombinedTopKModelSaver.get_parse_path(ori_config_path, _mode, folds=folds, shuffle=shuffle, seed=seed)
            need_retrain = True
            if os.path.exists(cv_path):
                try:
                    op_list_dict, estimator_dict, _ = CombinedTopKModelSaver._load(cv_path)
                    key = get_kfold_name(folds=folds, fold=fold, seed=seed, shuffle=shuffle)
                    op_list, estimator = op_list_dict[key], estimator_dict[key]
                    need_retrain = False
                except:
                    print("Read model failed, retain!")
            if need_retrain:
                try:
                    # train_node, op_list = parse_config(train_node, config, record=True, if_imbal=if_imbal)
                    # 用全数据做，效果好一点
                    try:
                        _, op_list = parse_config(data_node, config, record=True, if_imbal=if_imbal)
                        # op_list, _, _ = CombinedTopKModelSaver._load(ori_config_path)
                        train_node = construct_node(train_node.copy_(), op_list)
                        valid_node = construct_node(valid_node.copy_(), op_list)
                    except:
                        op_list = {}
                    x_p1, y_p1 = train_node.data
                    x_v1, y_v1 = valid_node.data
                    # 和下面直接fit的区别在于是否考虑数据分布不均匀的smote
                    estimator = fetch_predict_estimator(task_type, config['algorithm'], config, x_p1, y_p1, X_val=x_v1, y_val=y_v1,
                                                        weight_balance=train_node.enable_balance,
                                                        data_balance=train_node.data_balance, num_cpus=kwargs.get('num_cpus', 'auto'))
                except:
                    print("Training base model failed, use original model!")
                    op_list, estimator, _ = CombinedTopKModelSaver._load(ori_config_path)

            # print(model_idx, config['Autogluon_wraper:model_name'], time.time() - start)
        else:
            # start = time.time()
            n_base_model = kwargs['n_base_model']
            dropout = kwargs['dropout']

            if dropout > 0:
                dropout_num = int(n_base_model * dropout)
                if dropout_num > 0:
                    rng = np.random.default_rng(seed=1 + 1000 * layer + 100 * fold + model_idx)
                    # rng = np.random

                    # feature_dim = data_node.data[0].shape[1]
                    # n_dim = feature_dim // n_base_model
                    # for row in range(train_node.data[0].shape[0]):
                    #     dropout_idx = np.random.choice(n_base_model, size=dropout_num, replace=False)

                    #     for d in range(n_dim):
                    #         remain_mask = [-idx*n_dim-d-1 for idx in range(n_base_model) if idx not in dropout_idx]
                    #         for didx in dropout_idx:
                    #             train_node.data[0][row, -didx*n_dim-d-1] = train_node.data[0][row, remain_mask].mean()

                    train_num, all_dim = train_node.data[0].shape
                    predict_dim = data_node.data[0].shape[1]
                    ori_dim = all_dim - predict_dim
                    n_dim = predict_dim // n_base_model
                    dropout_mask = np.zeros((train_num, n_base_model), dtype=int)
                    for i in range(train_num):
                        dropout_mask[i, rng.choice(n_base_model, dropout_num, replace=False)] = 1

                    for dim in range(n_dim):
                        col =[ori_dim + dim + idx * n_dim for idx in range(n_base_model)]
                        if isinstance(train_node.data[0], np.ndarray):
                            data_pure = train_node.data[0][:, col] * (1 - dropout_mask)
                            train_node.data[0][:, col] = data_pure + dropout_mask * np.sum(data_pure, axis=1, keepdims=True) / (n_base_model - dropout_num)
                        elif isinstance(train_node.data[0], pd.DataFrame):
                            data_pure = train_node.data[0].iloc[:, col] * (1 - dropout_mask)
                            train_node.data[0].iloc[:, col] = data_pure + dropout_mask * np.sum(data_pure.values, axis=1, keepdims=True) / (n_base_model - dropout_num)

            # mid_time = time.time()

            if task_type in CLS_TASKS:
                _, estimator = get_cls_estimator(config, config['algorithm'])
            elif task_type in RGS_TASKS:
                _, estimator = get_rgs_estimator(config, config['algorithm'])
            else: raise ValueError("Wrong task type: %d" % task_type)

            _fit_params = {}
            if estimator.get_properties()['shortname'] == 'NN' or estimator.get_properties()['shortname'] == 'Autogluon wraper':
                _fit_params['X_val'] = valid_node.data[0]
                _fit_params['y_val'] = valid_node.data[1]
            estimator.get_data_info(data_node.feature_map)
            # estimator.fit(train_node.data[0], train_node.data[1], **_fit_params)
            try:
                estimator.fit(train_node.data[0], train_node.data[1], **_fit_params)
            except Exception as e:
                print("EEEEError", e)
                return False, None, None, None

            # print(mid_time - start, time.time() - mid_time)

        pred = fetch_predict_results(task_type, op_list, estimator, valid_node)
        if len(pred.shape) == 1: pred = pred.reshape(-1, 1)
        preds = {'train': pred}
        valid_nodes = kwargs.get('valid_nodes', {})
        for key in valid_nodes:
            _data_node = valid_nodes[key]
            tmp_node = _data_node.copy_()
            if ori_x is not None :
                tmp_node.data = (BaseEvaluator._concat(ori_x[key], _data_node.data[0]), _data_node.data[1])
                tmp_node.reset_feature_map()
            pred = fetch_predict_results(task_type, op_list, estimator, tmp_node)
            if len(pred.shape) == 1: pred = pred.reshape(-1, 1)
            preds[key] = pred

    return need_save, op_list, estimator, preds


def parallel_fit(config, task_type, if_imbal, seed,
                 layer, model_idx, fold, folds,
                 train_index, valid_index,
                 node_path, ori_x_path, output_dir, ori_config_path, cpu_ids, shuffle, **kwargs):

    p = psutil.Process()
    p.cpu_affinity(cpu_ids)
    # print(model_idx, cpu_ids)

    # The 0th layer needs feature engineering, and ori_config_path is not None
    assert (layer == 0) == (ori_config_path is not None)

    data_node = CombinedTopKModelSaver._load(node_path)
    ori_x = None
    if ori_x_path is not None and not isinstance(config, str):
        ori_x = CombinedTopKModelSaver._load(ori_x_path)

    valid_nodes = {}
    valid_paths = kwargs.pop('valid_paths', {})
    for key in valid_paths:
        valid_nodes[key] = CombinedTopKModelSaver._load(valid_paths[key])

    need_save, op_list, estimator, preds = _fit(config=config, task_type=task_type, if_imbal=if_imbal, seed=seed, 
                                                layer=layer,model_idx=model_idx, fold=fold, folds=folds,
                                                train_index=train_index, valid_index=valid_index,
                                                data_node=data_node, ori_x=ori_x, ori_config_path=ori_config_path, shuffle=shuffle, valid_nodes=valid_nodes, **kwargs)

    if preds is None:
        return model_idx, fold, None, need_save, cpu_ids

    if need_save:
        estimator_path = os.path.join(output_dir, get_estimator_name(layer, model_idx, fold))
        CombinedTopKModelSaver._save([op_list, estimator], estimator_path)

    return model_idx, fold, preds, need_save, cpu_ids


def parallel_predict(model_idx, config_path, task_type, node_path):

    op_list, model, _ = CombinedTopKModelSaver._load(config_path)
    valid_node = CombinedTopKModelSaver._load(node_path)
    y_valid_pred = fetch_predict_results(task_type, op_list, model, valid_node)

    return model_idx, y_valid_pred


def layer_fit(stack_configs, n_base_model, new_node, ori_xs, 
              task_type, if_imbal, seed, layer, output_dir, 
              thread=1, folds = 5, shuffle=True, ori_config_paths=None, logger=None, metric=None, skip_mask=None, mode='partial', val_nodes: dict=None, dropout=0):

    need_ori_xs = ori_xs is not None

    if len(stack_configs[0]) + len(stack_configs[1]) == 0:
        return None, None, None, None, None

    assert (layer == 0) == (ori_config_paths is not None)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # stack configs的数量
    _n_base = len(stack_configs[0])
    start = time.time()
    data_len = new_node.data[0].shape[0]
    n_dim = 1
    if task_type in CLS_TASKS:
        unique_num = len(np.unique(new_node.data[1]))
        if unique_num > 2:
            n_dim = unique_num

    if layer != 0:
        assert n_dim == new_node.data[0].shape[1] // n_base_model

    new_features = {'train': np.full((data_len, n_base_model * n_dim), np.nan)}
    if val_nodes is not None:
        for key in val_nodes:
            new_features[key] = np.zeros((val_nodes[key].data[0].shape[0], n_base_model * n_dim))
    head_outputs = dict()

    sms = [[None] * folds for _ in range(n_base_model)]
    ops = [[None] * folds for _ in range(n_base_model)]
    if skip_mask is not None:
        for i in range(n_base_model):
            if skip_mask[i]:
                sms[i] = None
                ops[i] = None

    train_indexes = []
    valid_indexes = []
    for train_index, valid_index in BaseEvaluator._get_cv_data(task_type, new_node, {"folds": folds, 'shuffle':shuffle}, seed, only_index=True):
        train_indexes.append(train_index)
        valid_indexes.append(valid_index)

    if thread == 1:
        for suc_cnt, config in enumerate(stack_configs[0] + stack_configs[1]):
            if suc_cnt < _n_base and skip_mask is not None and skip_mask[suc_cnt]:
                continue
            ori_xs = None
            if need_ori_xs and suc_cnt < _n_base:
                ori_xs = CombinedTopKModelSaver._load(os.path.join(output_dir, get_ori_x_name(suc_cnt)))
            model_idx = suc_cnt
            for fold in range(1, folds+1):
                _fold = fold

                ori_config_path = None if ori_config_paths is None else ori_config_paths[suc_cnt]
                kwargs = {
                    'config': config, 'task_type': task_type, 'if_imbal': if_imbal, 'seed': seed, 
                    'layer': layer, 'model_idx': model_idx, 'fold': _fold, 'folds': folds,
                    'train_index': train_indexes[fold-1], 'valid_index': valid_indexes[fold-1],
                    'data_node': new_node, 'ori_x': ori_xs, 'ori_config_path': ori_config_path, 'n_base_model': n_base_model, 'shuffle': shuffle
                }
                if suc_cnt >= _n_base:
                    kwargs.update({'metric': metric})
                else:
                    kwargs.update({'valid_nodes': val_nodes})
                    if layer == 0:
                        kwargs.update({'mode': mode})
                    else:
                        kwargs.update({'dropout': dropout})

                need_save, op_list, estimator, preds = _fit(**kwargs)
                if model_idx < _n_base:
                    if preds is not None:
                        new_features['train'][valid_indexes[_fold-1], model_idx * n_dim:(model_idx + 1) * n_dim] = preds['train'][:, -n_dim:]
                        if val_nodes is not None:
                            for key in val_nodes.keys():
                                new_features[key][:, model_idx * n_dim:(model_idx + 1) * n_dim] += preds[key][:, -n_dim:] / folds
                        if need_save:
                            sms[model_idx][_fold-1] = estimator
                            ops[model_idx][_fold-1] = op_list
                    else:
                        for key in val_nodes.keys():
                            new_features[key][:, model_idx * n_dim:(model_idx + 1) * n_dim] = np.nan
                else:
                    pred = preds['train']
                    _config = stack_configs[1][model_idx-_n_base]
                    head = f"{_config}-L{layer}"
                    if head not in head_outputs:
                        head_outputs[head] = np.zeros((data_len,) + pred.shape[1:])
                    head_outputs[head][valid_indexes[_fold-1]] = pred  

    else:
        p = psutil.Process()
        all_cpus = p.cpu_affinity()
        assert len(all_cpus) >= thread, f"The number of available cpus{len(all_cpus)} must not be smaller than the number of threads({thread})!"

        cpu_per_sub = len(all_cpus)
        available_cpus = all_cpus

        # allocate cpu cores for each model
        need_cpus = []
        for conf in stack_configs[0]:
            need_cpu = cpu_per_sub
            if conf['algorithm'] == 'Autogluon_wraper' and conf['Autogluon_wraper:model_name'] in ['FASTAI', 'NN_TORCH']:
                need_cpu = len(all_cpus)
                # need_cpu = cpu_per_sub
            need_cpus.append(need_cpu)
        need_cpus += [1] * len(stack_configs[1])
        cpu_idxes = np.argsort(-np.array(need_cpus))

        node_path = os.path.join(output_dir, get_node_name(layer, 'train'))
        with open(node_path, 'wb') as f:
            pkl.dump(new_node, f)
        valid_paths = {}
        if val_nodes is not None:
            for key in val_nodes:
                path = os.path.join(output_dir, get_node_name(layer, key))
                with open(path, 'wb') as f:
                    pkl.dump(val_nodes[key], f)
                valid_paths[key] = path

        with cfutures.ProcessPoolExecutor(max_workers=thread) as executor:
            fs_wait = set()
            for tmp_idx in cpu_idxes:
                suc_cnt = tmp_idx
                config = (stack_configs[0] + stack_configs[1])[tmp_idx]

                if suc_cnt < _n_base and skip_mask is not None and skip_mask[suc_cnt]:
                    continue

                for fold in range(1, folds+1):

                    ori_config_path = None if ori_config_paths is None else ori_config_paths[suc_cnt]
                    kwargs = {
                        'config': config, 'task_type': task_type, 'if_imbal': if_imbal, 'seed': seed, 
                        'layer': layer, 'model_idx': suc_cnt, 'fold': fold, 'folds': folds,
                        'train_index': train_indexes[fold-1], 'valid_index': valid_indexes[fold-1],
                        'node_path': node_path, 'ori_x_path': None if not need_ori_xs else os.path.join(output_dir, get_ori_x_name(suc_cnt)),
                        'output_dir': output_dir, 'ori_config_path': ori_config_path, 'n_base_model': n_base_model, 'shuffle': shuffle
                    }
                    if suc_cnt >= _n_base:
                        kwargs.update({'metric': metric})
                    else:
                        kwargs.update({'valid_paths': valid_paths})
                        if layer == 0:
                            kwargs.update({'mode': mode})
                        else:
                            kwargs.update({'dropout': dropout})

                    if len(fs_wait) < thread and len(available_cpus) >= need_cpus[suc_cnt]:
                        kwargs['cpu_ids'] = [available_cpus.pop() for _ in range(need_cpus[suc_cnt])]
                        fs_wait.add(executor.submit(parallel_fit, **kwargs))
                    else:
                        fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
                        first_ok = True
                        for fi, fs in enumerate(fs_done):
                            model_idx, _fold, preds, need_save, cpu_ids = fs.result()
                            available_cpus.extend(cpu_ids)

                            if first_ok and len(available_cpus) >= need_cpus[suc_cnt]:
                                kwargs['cpu_ids'] = [available_cpus.pop() for _ in range(need_cpus[suc_cnt])]
                                fs_wait.add(executor.submit(parallel_fit, **kwargs))
                                first_ok = False
                            if model_idx < _n_base:
                                if preds is not None:
                                    new_features['train'][valid_indexes[_fold-1], model_idx * n_dim:(model_idx + 1) * n_dim] = preds['train'][:, -n_dim:]
                                    if val_nodes is not None:
                                        for key in val_nodes.keys():
                                            new_features[key][:, model_idx * n_dim:(model_idx + 1) * n_dim] += preds[key][:, -n_dim:] / folds
                                    if need_save:
                                        estimator_path = os.path.join(output_dir, get_estimator_name(layer, model_idx, _fold))
                                        op_list, estimator = CombinedTopKModelSaver._load(estimator_path)
                                        sms[model_idx][_fold-1] = estimator
                                        ops[model_idx][_fold-1] = op_list
                                        os.remove(estimator_path)
                                else:
                                    for key in val_nodes.keys():
                                        new_features[key][:, model_idx * n_dim:(model_idx + 1) * n_dim] = np.nan
                            else:
                                pred = preds['train']
                                _config = stack_configs[1][model_idx-_n_base]
                                head = f"{_config}-L{layer}"
                                if head not in head_outputs:
                                    head_outputs[head] = np.zeros((data_len,) + pred.shape[1:])
                                head_outputs[head][valid_indexes[_fold-1]] = pred

            while len(fs_wait) > 0:
                fs_done, fs_wait = cfutures.wait(fs_wait, return_when=cfutures.FIRST_COMPLETED)
                for fs in fs_done:
                    model_idx, _fold, preds, need_save, cpu_ids = fs.result()
                    available_cpus.extend(cpu_ids)
                    if model_idx < _n_base:
                        if preds is not None:
                            new_features['train'][valid_indexes[_fold-1], model_idx * n_dim:(model_idx + 1) * n_dim] = preds['train'][:, -n_dim:]
                            if val_nodes is not None:
                                for key in val_nodes.keys():
                                    new_features[key][:, model_idx * n_dim:(model_idx + 1) * n_dim] += preds[key][:, -n_dim:] / folds
                            if need_save:
                                estimator_path = os.path.join(output_dir, get_estimator_name(layer, model_idx, _fold))
                                op_list, estimator = CombinedTopKModelSaver._load(estimator_path)
                                sms[model_idx][_fold-1] = estimator
                                ops[model_idx][_fold-1] = op_list
                                os.remove(estimator_path)
                        else:
                            for key in val_nodes.keys():
                                new_features[key][:, model_idx * n_dim:(model_idx + 1) * n_dim] = np.nan
                    else:
                        pred = preds['train']
                        _config = stack_configs[1][model_idx-_n_base]
                        head = f"{_config}-L{layer}"
                        if head not in head_outputs:
                            head_outputs[head] = np.zeros((data_len,) + pred.shape[1:])
                        head_outputs[head][valid_indexes[_fold-1]] = pred

        os.remove(node_path)
        for path in valid_paths.values():
            os.remove(path)

    cost = time.time() - start
    print(f"Cost of Layer{layer} training with {thread} threads: {cost}s")

    # save cv models for 0th layer
    if layer == 0:
        logger.info("Start to save cv model!")
        for i, ori_config_path in enumerate(ori_config_paths):
            if skip_mask is not None and skip_mask[i]:
                continue
            _mode = 'cv'
            if mode == 'partial':
                _mode += mode
            cv_path = CombinedTopKModelSaver.get_parse_path(ori_config_path, _mode, folds=folds, shuffle=shuffle, seed=seed)
            if not os.path.exists(cv_path):
                logger.info("Save cv model [%s], path: %s" % (stack_configs[0][i]['algorithm'], cv_path))
                op_list_dict = dict()
                estimator_dict = dict()
                for fold in range(1, folds+1):
                    assert sms[i][fold-1] is not None and ops[i][fold-1] is not None
                    key = get_kfold_name(folds=folds, fold=fold, seed=seed, shuffle=shuffle)
                    op_list_dict[key] = ops[i][fold-1]
                    estimator_dict[key] = sms[i][fold-1]
                CombinedTopKModelSaver._save([op_list_dict, estimator_dict, None], cv_path)

    # check whether some models fail to train
    for i in range(len(sms)):
        sm = sms[i]
        if sm is not None:
            fail = False
            for j in range(len(sm)):
                if sm[j] is None:
                    fail = True

            if fail:
                sms[i] = None
                ops[i] = None

    return sms, ops, new_features, head_outputs, cost


def save_ori_x(ori_xs, output_dir):

    if ori_xs is not None:
        for suc_cnt in range(len(ori_xs['train'])):
            ori_x = {key: ori_xs[key][suc_cnt] for key in ori_xs}
            ori_x_path = os.path.join(output_dir, get_ori_x_name(suc_cnt))
            if not os.path.exists(ori_x_path):
                CombinedTopKModelSaver._save(ori_x, ori_x_path)

def rm_ori_x(ori_xs, output_dir):

    if ori_xs is not None:
        for suc_cnt in range(len(ori_xs['train'])):
            ori_x_path = os.path.join(output_dir, get_ori_x_name(suc_cnt))
            if os.path.exists(ori_x_path):
                os.remove(ori_x_path)