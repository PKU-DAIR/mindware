from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import *
from mindware.modules.autogluon_zeroshot.utils import *
from mindware.modules.autogluon_zeroshot.zeroshot_portfolio_2023 import hyperparameter_portfolio_zeroshot_2023
import datetime
from mindware.components.optimizers.base_optimizer import MAX_INT
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.modules.base import BaseAutoML
from openbox.utils.limit import run_obj_func
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from mindware.utils.logging_utils import setup_logger, get_logger
from openbox.utils.history import Observation
import os
import time
import numpy as np

class Zeroshot(BaseAutoML):
    name ='zeroshot'
    def __init__(self, task_type,
                 metric, data_node,
                 time_limit,
                 evaluation = 'holdout',
                 per_run_time_limit=600,
                 output_dir='./data',
                 seed = 1,
                 topk = 50,
                 amount_of_resource=None,
                 task_id = 'task_id'
                 ):
        super().__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=None,
            optimizer=None, inner_iter_num_per_iter=1,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=1, topk=1000, rmfiles=None,
            ensemble_method=None, ensemble_size=0, task_id=task_id
        )

        path = 'AutoGluon-%s_%s_%s' % (
            self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.metric_name = metric
        self.metric = get_metric(metric)
        self.task_type = task_type
        self.problem_type = transform_mindware2autogluon_tasktype(task_type)
        self.evaluation = evaluation

        self.data_node = data_node

        self.time_limit = time_limit
        self.per_run_time_limit = per_run_time_limit
        self.seed = seed
        self.topk = topk
        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource

        self.configs = trans_portfolio_to_params_list(hyperparameter_portfolio_zeroshot_2023, self.problem_type)
        self.logger = self._get_logger()
        self.evaluator = None
        self.reshuffle_ratio = 0

        if self.task_type in CLS_TASKS:
            from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
            self.evaluator = CASHCLSEvaluator(
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
            from mindware.modules.cash.cash_evaluator import CASHRGSEvaluator
            self.evaluator = CASHRGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                reshuffle_ratio=self.reshuffle_ratio)

        self.now_fit_alg_number = 0
        self.history = []


        self.topk_saver = CombinedTopKModelSaver(
            k=topk, model_dir=self.output_dir,
            identifier=datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        )

    def _get_logger(self):
        logger_name = 'Zeroshot-(%d)' % (self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)


    def iterate(self, trail_num=None):

        budget = self.per_run_time_limit


        model_name, params = self.configs[self.now_fit_alg_number]

        if 'ag_args' in params:
            params.pop('ag_args')
        config = dict()
        config['algorithm'] = 'Autogluon_wraper'
        config['Autogluon_wraper:model_name'] = model_name
        config['Autogluon_wraper:eval_metric'] = self.metric_name
        config['Autogluon_wraper:problem_type'] = self.problem_type
        config['Autogluon_wraper:params'] = params

        self.now_fit_alg_number += 1
        obj_args, obj_kwargs = (config, ), dict()

        result = run_obj_func(self.evaluator, obj_args, obj_kwargs, timeout=budget)

        ret, timeout_status, traceback_msg, elapsed_time = (
        result['result'], result['timeout'], result['traceback'], result['elapsed_time'])

        if timeout_status:
            trial_state = TIMEOUT
        elif traceback_msg is not None:
            trial_state = FAILED
            self.logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
        else:
            trial_state = SUCCESS
        if trial_state == SUCCESS:
            objectives, constraints, extra_info = parse_result(ret)
        else:
            objectives, constraints, extra_info = np.inf, None, None
            # update observation to advisor

        if self.now_fit_alg_number >= len(self.configs):
            self.early_stop_flag = True

        if time.time() - self.timestamp >= self.time_limit:
            self.timeout_flag=True
            if self.timeout_flag:
                self.logger.info(f"Time out({self.time_limit}s)!")

        if objectives == np.inf:
            return self.incumbent

        observation = Observation(
                config=config, objectives=objectives, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
        )
        self.update_saver([config], [objectives[0]])
        self.history.append(observation)
        self.logger.info('Iter %d, objectives: %s.' % (self.now_fit_alg_number, objectives))

        self.incumbent_perf = min(self.history, key= lambda X:X.objectives).objectives
        self.incumbent = min(self.history, key= lambda X:X.objectives).config
        return self.incumbent


    def update_saver(self, config_list, perf_list):
        # perf_list: perf - the smaller, the better
        # Check if all the configs is valid in case of storing None into the config file
        all_invalid = True

        for i, perf in enumerate(perf_list):
            if np.isfinite(perf) and perf != MAX_INT:
                all_invalid = False
                if not isinstance(config_list[i], dict):
                    config = config_list[i].get_dictionary().copy()
                else:
                    config = config_list[i].copy()
                if self.evaluator.fixed_config is not None:
                    if not isinstance(self.evaluator.fixed_config, dict):
                        fixed_config = self.evaluator.fixed_config.get_dictionary().copy()
                    else:
                        fixed_config = self.evaluator.fixed_config.copy()
                    config.update(fixed_config)
                if 'algorithm' not in config:
                    assert 'ensemble_size' in config
                    classifier_id = 'ens'
                else:
                    classifier_id = config['algorithm']
                # -perf: The larger, the better.
                save_flag, model_path, delete_flag, model_path_deleted = self.topk_saver.add(config, -perf,
                                                                                             classifier_id)
                # By default, the evaluator has already stored the models.
                if self.evaluation in ['holdout', 'partial', 'partial_bohb', 'cv']:
                    if save_flag or self.name == 'ens':
                        pass
                    else:
                        if os.path.exists(model_path):
                            os.remove(model_path)
                            self.logger.info("Model deleted from %s" % model_path)
                        else:
                            self.logger.error("Model path %s does not exist!" % model_path)
                    try:
                        if delete_flag:
                            os.remove(model_path_deleted)
                            self.logger.info("Model deleted from %s" % model_path_deleted)
                        else:
                            pass
                    except:
                        pass
            else:
                continue

        if not all_invalid:
            self.topk_saver.save_topk_config()
