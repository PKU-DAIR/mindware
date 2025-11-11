import os

from typing import Union, Callable
from sklearn.metrics._scorer import _BaseScorer

from mindware.modules.base import BaseAutoML

from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.config_space.cs_builder import get_hpo_cs
import json


class BaseHPO(BaseAutoML):
    
    name='hpo'
    
    def __init__(self, estimator_id: str, task_type: str = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac',
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test', reshuffle_ratio=0):

        super(BaseHPO, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=1,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id
        )

        if optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.estimator_id = estimator_id

        path = 'HPO(%s)-%s(%d)-%s_%s_%s' % (
            self.estimator_id, optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_cs_args
        cs_args = get_cs_args(**cs_args)
        self.cs_args = cs_args
        # _candidates = None
        self.cs = get_hpo_cs(self.estimator_id, self.task_type, **cs_args)

        # Define evaluator and optimizer
        self.evaluator = None
        self.reshuffle_ratio = reshuffle_ratio
        if self.task_type in CLS_TASKS:
            from mindware.modules.hpo.hpo_evaluator import HPOCLSEvaluator
            self.evaluator = HPOCLSEvaluator(
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
            from mindware.modules.hpo.hpo_evaluator import HPORGSEvaluator
            self.evaluator = HPORGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                reshuffle_ratio=self.reshuffle_ratio)

        self.optimizer = self.build_optimizer(name='hpo')

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-HPO-%s-%s(%d)' % (self.estimator_id, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseHPO, self).get_conf()
        conf['estimator_id'] = self.estimator_id

        if save:
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf, f, indent=4)

        return conf
