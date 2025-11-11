from mindware.components.config_space.cls_cs_builder import get_cash_cs as get_cls_cash_cs
from mindware.components.config_space.rgs_cs_builder import get_cash_cs as get_rgs_cash_cs

from mindware.components.config_space.cls_cs_builder import get_fe_cs as get_cls_fe_cs
from mindware.components.config_space.rgs_cs_builder import get_fe_cs as get_rgs_fe_cs
from mindware.components.utils.constants import *

from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons as _cls_addons
from mindware.components.models.regression import _regressors, _addons as _rgs_addons

from ConfigSpace import ConfigurationSpace, Constant, CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, UnParametrizedHyperparameter
from ConfigSpace.forbidden import ForbiddenInClause, ForbiddenAndConjunction, ForbiddenEqualsClause

import numpy as np


def get_cs_args(**kwargs):
    resampling_params = kwargs.get('resampling_params', None)
    data_node = kwargs.get('data_node', None)

    test_size = 0.33
    if resampling_params is not None and 'test_size' in resampling_params:
        test_size = resampling_params['test_size']

    _cs_args = {}
    if data_node is not None:
        n_samples = int(data_node.data[0].shape[0] * (1 - test_size))
        _cs_args = {
            'n_classes': np.unique(data_node.data[1]).shape[0],
            'n_features': data_node.data[0].shape[1],
            'n_samples': n_samples,
            'y_neg_mask': np.all(data_node.data[1] >= 0)
        }
    
    return _cs_args

def get_fe_cs_args(**kwargs):
    _cs_args = get_cs_args(**kwargs)
    
    resampling_params = kwargs.get('resampling_params', None)
    data_node = kwargs.get('data_node', None)

    test_size = 0.33
    if resampling_params is not None and 'test_size' in resampling_params:
        test_size = resampling_params['test_size']

    if data_node is not None:
        max_zero_ratio = np.max(np.sum(data_node.data[0] == 0, axis=0) / data_node.data[0].shape[0])
        _cs_args['zero_ratio_mask'] = max_zero_ratio < (1 - test_size)
        from sklearn.decomposition import FastICA
        try:
            ica = FastICA(fun='exp', algorithm='deflation', whiten=False)
            ica.fit(data_node.data[0][:10000, :50])
            _cs_args['exp_deflation_mask'] = True
        except:
            _cs_args['exp_deflation_mask'] = False
    
        try:
            ica = FastICA(fun='cube', algorithm='parallel', whiten=False)
            ica.fit(data_node.data[0])
            _cs_args['cube_parallel_mask'] = True
        except:
            _cs_args['cube_parallel_mask'] = False
        
    return _cs_args


def get_cash_cs(include_algorithms=None, task_type=CLASSIFICATION, **cs_args):

    if task_type in CLS_TASKS:
        cs = get_cls_cash_cs(include_algorithms, task_type, **cs_args)
    else:
        cs = get_rgs_cash_cs(include_algorithms, task_type, **cs_args)

    return cs


def get_hpo_cs(estimator_id, task_type, **cs_args):

    if task_type in CLS_TASKS:
        _candidates = get_combined_candidtates(_classifiers, _cls_addons)
    else:
        _candidates = get_combined_candidtates(_regressors, _rgs_addons)

    if estimator_id not in _candidates:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    
    cash_cs = get_cash_cs(include_algorithms=[estimator_id], task_type=task_type, **cs_args)

    cs = ConfigurationSpace()
    cs.add_hyperparameter(Constant('algorithm', estimator_id))
    # Add active hyperparameters
    hps = cash_cs.get_hyperparameters()
    for hp in hps:
        if hp.name.split(':')[0] == estimator_id:
            cs.add_hyperparameter(hp)
    # Add active conditions
    conds = cash_cs.get_conditions()
    for cond in conds:
        try:
            cs.add_condition(cond)
        except:
            pass
    # Add active forbidden clauses
    forbids = cash_cs.get_forbiddens()
    for forbid in forbids:
        try:
            cs.add_forbidden_clause(forbid)
        except:
            pass

    return cs


def get_fe_cs(task_type=CLASSIFICATION, include_image=False,
              include_text=False, include_preprocessors=None, if_imbal=False, **cs_args):

    if task_type in CLS_TASKS:
        cs = get_cls_fe_cs(task_type, include_image=include_image, include_text=include_text,
                           include_preprocessors=include_preprocessors, if_imbal=if_imbal, **cs_args)
    else:
        cs = get_rgs_fe_cs(task_type, include_image=include_image, include_text=include_text,
                           include_preprocessors=include_preprocessors, **cs_args)

    return cs


def get_ens_cs(**cs_args):
    layer_upper = cs_args.get('layer_upper', 1)
    size_upper = cs_args.get('size_upper', 40)

    size_def = cs_args.get('size_default', 10)
    ratio_def = cs_args.get('ratio_default', 40)
    dropout_def = cs_args.get('dropout_default', 0)

    cs = ConfigurationSpace()

    meta_learner = CategoricalHyperparameter('meta_learner', ['weighted', 'linear', 'best'])
    stack_layers = UniformIntegerHyperparameter('stack_layers', 0, layer_upper, default_value=0)

    ensemble_size = UniformIntegerHyperparameter('ensemble_size', 2, size_upper, default_value=size_def, q=2)
    ratio = UniformIntegerHyperparameter("ratio", 0, 48, default_value=ratio_def, q=4)
    dropout = Constant("dropout", dropout_def)
    # dropout = UniformIntegerHyperparameter("dropout", 0, 20, default_value=dropout_def, q=20)

    cs.add_hyperparameters([meta_learner, stack_layers, ensemble_size, ratio, dropout])

    return cs
