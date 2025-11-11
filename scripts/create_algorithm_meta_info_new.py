import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# 将当前文件所在文件夹的上层目录加入到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.components.utils.constants import MULTICLASS_CLS, BINARY_CLS, REGRESSION, CLS_TASKS, RGS_TASKS, CATEGORICAL
from mindware.components.metrics.metric import get_metric
from mindware.utils.data_manager import DataManager
from mindware.components.feature_engineering.transformation_graph import DataNode

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/root/automl_data/automl_data')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--datasets', type=str, default='diabetes')
parser.add_argument('--metrics', type=str, default='all')
parser.add_argument('--task', type=str, choices=['reg', 'cls'], default='cls')
parser.add_argument('--algo', type=str, default='all')
parser.add_argument('--time_limit', type=int, default=1200)
parser.add_argument('--amount_of_resource', type=int, default=100)
args = parser.parse_args()

base_dir = './data_rgs'
if args.task == 'cls':
    base_dir = './data_cls'

datasets = args.datasets.split(',')
start_id, rep = args.start_id, args.rep
time_limit = args.time_limit
amount_of_resource = args.amount_of_resource
save_dir = os.path.join(base_dir, 'meta_res/')
data_dir = args.data_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# cls_metrics = ['acc', 'f1', 'auc']
reg_metrics = ['mse', 'r2', 'mae']

# cls_metrics = ['acc']
# reg_metrics = ['mse']

# cls_metrics = ['f1', 'auc']
# reg_metrics = ['r2', 'mae']

cls_metrics = ['cls_mse']

def load_data(dataset, data_dir='./', datanode_returned=False, preprocess=True, task_type=None):
    dm = DataManager()
    if task_type is None:
        data_path = os.path.join(data_dir, 'datasets/%s.csv' % dataset)
    elif task_type in CLS_TASKS:
        data_path = os.path.join(data_dir, 'cls_datasets/%s.csv' % dataset)
    elif task_type in RGS_TASKS:
        data_path = os.path.join(data_dir, 'rgs_datasets/%s.csv' % dataset)
    else:
        raise ValueError("Unknown task type %s" % str(task_type))

    # Load train data.
    if dataset in ['higgs', 'amazon_employee', 'spectf', 'usps', 'vehicle_sensIT', 'codrna']:
        label_column = 0
    elif dataset in ['rmftsa_sleepdata(1)']:
        label_column = 1
    else:
        label_column = -1

    if dataset in ['spambase', 'messidor_features']:
        header = None
    else:
        header = 'infer'

    if dataset in ['winequality_white', 'winequality_red']:
        sep = ';'
    else:
        sep = ','

    train_data_node = dm.load_train_csv(data_path, label_col=label_column, header=header, sep=sep,
                                        na_values=["n/a", "na", "--", "-", "?"])

    if preprocess:
        train_data = dm.preprocess_fit(train_data_node, task_type)
    else:
        train_data = train_data_node

    if datanode_returned:
        return train_data, dm
    else:
        X, y = train_data.data
        feature_types = train_data.feature_types
        return X, y, feature_types, dm


def load_train_test_data(dataset, data_dir='./', test_size=0.2, task_type=None, random_state=45):
    X, y, feature_type, dm = load_data(dataset, data_dir, False, task_type=task_type)
    if task_type is None or task_type in CLS_TASKS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    train_node = DataNode(data=[X_train, y_train], feature_type=feature_type.copy())
    test_node = DataNode(data=[X_test, y_test], feature_type=feature_type.copy())
    # print('is imbalanced dataset', is_imbalanced_dataset(train_node))
    return train_node, test_node, dm


def evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, time_limit=600, amount_of_resource=100, seed=1, task_type=None):

    _save_dir = os.path.join(save_dir, obj_metric)
    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)
    save_path = _save_dir + '%s-%s-%s-%d-%d.pkl' % (dataset, algo, obj_metric, run_id, time_limit)
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            res = pickle.load(f)
        eval_num = res[2]
        if eval_num >= 90:
            return

    _algo = [algo]
    print('EVALUATE-%s-%s-%s: run_id=%d' % (dataset, algo, obj_metric, run_id))
    train_data, test_data, dm = load_train_test_data(dataset, data_dir=data_dir, task_type=task_type)
    if task_type in CLS_TASKS:
        task_type = BINARY_CLS if len(set(train_data.data[1])) == 2 else MULTICLASS_CLS
    # print(set(train_data.data[1]))

    from mindware import CASHFE
    opt = CASHFE(
                include_algorithms=_algo, sub_optimizer='smac', task_type=task_type,
                metric=obj_metric,
                data_node=train_data, evaluation='holdout', resampling_params={'test_size': 0.33},
                optimizer='block_0', inner_iter_num_per_iter=1,
                time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=180,
                output_dir=os.path.join(base_dir, f'data/{obj_metric}'), seed=int(seed), n_jobs=1, topk=amount_of_resource, rmfiles=True,
                ensemble_method=None, task_id=dataset
            )

    print(opt.get_conf(save=True))

    print(opt.run())
    print(opt.get_model_info(save=True))

    validation_score = opt.incumbent_perf
    eval_num = len(opt.optimizer.perfs)

    scorer = opt.metric
    if obj_metric in ['cls_mse', 'auc']:
        pred = opt.predict_proba(test_data, ens=False)
        if obj_metric == 'auc':
            pred = pred[:, 1]
    else:
        pred = opt.predict(test_data, ens=False)
    test_score = scorer._score_func(test_data.data[1], pred) * scorer._sign

    save_path = os.path.join(_save_dir, '%s-%s-%s-%d-%d.pkl' % (dataset, algo, obj_metric, run_id, time_limit))
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, algo, eval_num, validation_score, test_score, task_type], f)


def check_datasets(datasets, task_type=None):
    for _dataset in datasets:
        try:
            _, _, _ = load_train_test_data(_dataset, data_dir=data_dir, random_state=1, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


if __name__ == "__main__":
    algorithms = [
                    'adaboost',
                    'extra_trees',
                    'gradient_boosting',
                    'k_nearest_neighbors',
                    'lda',
                    'liblinear_svc',
                    'libsvm_svc',
                    'lightgbm',
                    'logistic_regression',
                    'qda',
                    'random_forest',
                    'xgboost'
    ]
    task_type = MULTICLASS_CLS
    if args.task == 'reg':
        task_type = REGRESSION
        algorithms = [
                    'adaboost',
                    'extra_trees',
                    'gradient_boosting',
                    'k_nearest_neighbors',
                    'lasso_regression',
                    'liblinear_svr',
                    'libsvm_svr',
                    'lightgbm',
                    'random_forest',
                    'ridge_regression',
                    'xgboost'
    ]

    if args.algo != 'all':
        algorithms = args.algo.split(',')

    metrics = cls_metrics if args.task == 'cls' else reg_metrics
    if args.metrics != 'all':
        metrics = args.metrics.split(',')

    check_datasets(datasets, task_type=task_type)
    running_info = list()
    log_filename = 'running-%d.txt' % os.getpid()

    for dataset in datasets:
        for obj_metric in metrics:
            _save_dir = os.path.join(save_dir, obj_metric)
            np.random.seed(1)
            seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
            for algo in algorithms:
                for run_id in range(start_id, start_id + rep):
                    seed = seeds[run_id]
                    try:
                        task_id = '%s-%s-%s-%d: %s' % (dataset, algo, obj_metric, run_id, 'success')
                        evaluate_ml_algorithm(dataset, algo, run_id, obj_metric, time_limit=time_limit, amount_of_resource=amount_of_resource,
                                              seed=seed, task_type=task_type)
                    except Exception as e:
                        task_id = '%s-%s-%s-%d: %s' % (dataset, algo, obj_metric, run_id, str(e))
                        running_info.append(task_id)

                    print(task_id)
                    with open(os.path.join(_save_dir, log_filename), 'a') as f:
                        f.write('\n' + task_id)

            # Write down the error info.
            if len(running_info) > 0:
                with open(os.path.join(_save_dir, 'failed-%s' % log_filename), 'w') as f:
                    f.write('\n'.join(running_info))
