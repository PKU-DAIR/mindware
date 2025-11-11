import os
import json
from typing import List, Union, Callable
from copy import deepcopy

from sklearn.metrics._scorer import _BaseScorer

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode
import time
import datetime


class BaseCASHFE(BaseAutoML):
    
    name='cashfe'
    
    def __init__(self, include_algorithms: List[str] = None, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', sub_optimizer: str = 'smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5,
                 include_preprocessors=None, task_id='test',
                 filter_params=None, reshuffle_ratio=0):
        
        super(BaseCASHFE, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=inner_iter_num_per_iter,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id
        )

        if optimizer not in ['smac', 'tpe', 'random_search', 'mab', 'block_0', 'block_1', 'block_2', 'block_3', 'block_4']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.include_algorithms = include_algorithms
        path = 'CASHFE-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_fe_cs_args
        cs_args = get_fe_cs_args(**cs_args)
        self.cs_args = cs_args

        self.filter_params = filter_params
        # select models
        if self.filter_params is not None and 'n_algorithm' in self.filter_params:
            n_algo = self.filter_params['n_algorithm']
            include_algorithms = self._recommand_models(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_algo=n_algo, include_algorithms=include_algorithms)
    
        if include_algorithms is not None and len(include_algorithms) == 1:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            self.cs = get_hpo_cs(estimator_id=include_algorithms[0], task_type=self.task_type, **cs_args)
        else:
            from mindware.components.config_space.cs_builder import get_cash_cs
            self.cs = get_cash_cs(include_algorithms=include_algorithms, task_type=self.task_type, **cs_args)
            include_algorithms = list(self.cs['algorithm'].choices)
        # select preprocessors
        from mindware.components.config_space.cs_builder import get_fe_cs
        include_preprocessors_dict = {algo: include_preprocessors for algo in include_algorithms}
        if self.filter_params is not None and 'n_preprocessor' in self.filter_params:
            n_prep = self.filter_params['n_preprocessor']
            include_preprocessors_dict = self._recommand_preps(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_prep=n_prep, include_algorithms=include_algorithms, include_preprocessors=include_preprocessors)
            include_preprocessors = []
            for preps in include_preprocessors_dict.values():
                include_preprocessors.extend(preps)
            include_preprocessors = list(set(include_preprocessors))
        self.include_preprocessors_dict = include_preprocessors_dict

        fe_config_space_dict = {}
        for algo in include_algorithms:
            fe_config_space_dict[algo] = get_fe_cs(self.task_type, include_preprocessors=include_preprocessors_dict[algo], if_imbal=self.if_imbal, **cs_args)
        if self.optimizer_name != 'mab' and not self.optimizer_name.startswith('block'):
            tmp_cs = get_fe_cs(self.task_type, include_preprocessors=include_preprocessors, if_imbal=self.if_imbal, **cs_args)
            self.cs.add_hyperparameters(tmp_cs.get_hyperparameters())
            self.cs.add_conditions(tmp_cs.get_conditions())
            self.cs.add_forbidden_clauses(tmp_cs.get_forbiddens())

        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')

        # Define evaluator and optimizer
        self.evaluator = None
        self.reshuffle_ratio = reshuffle_ratio
        if self.task_type in CLS_TASKS:
            from mindware.modules.cashfe.cashfe_evaluator import CASHFECLSEvaluator
            self.evaluator = CASHFECLSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                if_imbal=self.if_imbal, 
                reshuffle_ratio=self.reshuffle_ratio)
        else:
            from mindware.modules.cashfe.cashfe_evaluator import CASHFERGSEvaluator
            self.evaluator = CASHFERGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                reshuffle_ratio=self.reshuffle_ratio)

        self.optimizer = self.build_optimizer(name='cashfe', sub_optimizer=sub_optimizer, fe_config_space_dict=fe_config_space_dict)

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASHFE-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseCASHFE, self).get_conf()
        from ConfigSpace.hyperparameters import Constant
        if isinstance(self.cs['algorithm'], Constant):
            conf['include_algorithms'] = [self.cs['algorithm'].value]
        else:
            conf['include_algorithms'] = self.cs['algorithm'].choices

        conf['include_preprocessors'] = self.include_preprocessors_dict

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf