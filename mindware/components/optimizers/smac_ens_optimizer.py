# License: MIT

import time
from typing import List
from tqdm import tqdm
import numpy as np
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.config_space.space_utils import get_config_from_dict
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.advisor import MyAdvisor
from mindware.modules.ens.ens_utils import better_ens
from copy import deepcopy
from mindware.modules.ens.ens_evaluator import EnsEvaluator
from ConfigSpace import Constant


class SMACEnsOptimizer(BaseOptimizer):

    def __init__(
            self, evaluator: EnsEvaluator, config_space, name, eval_type,
            time_limit=None, evaluation_limit=None,
            per_run_time_limit=300, per_run_mem_limit=1024, 
            inner_iter_num_per_iter=1, timestamp=None, 
            logging_dir='logs',
            task_id='OpenBox',
            output_dir='./', seed=1, n_jobs=1, topk=50):
        super(SMACEnsOptimizer, self).__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type, 
                                            time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                            per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                            inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                            output_dir=output_dir, seed=seed, topk=topk)
        self.FAILED_PERF = [np.inf]
        self.evaluator = evaluator
        self.per_run_time_limit = per_run_time_limit
        self.config_space = config_space
        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        self.config_advisor = MyAdvisor(config_space,
                                    initial_trials=4,
                                    init_strategy='latin_hypercube',  # default， latin_hypercube
                                    rand_prob=0.2,
                                    surrogate_type='gp',
                                    acq_type='ei',
                                    task_id=task_id,
                                    output_dir=logging_dir,
                                    random_state=self.seed,
                                    logger_kwargs=_logger_kwargs)

        self.RATIO_RANGE = (config_space['ratio'].lower, config_space['ratio'].upper)
        self.RQ = config_space['ratio'].q
        self.SIZE_RANGE = (config_space['ensemble_size'].lower, config_space['ensemble_size'].upper)
        self.SQ = config_space['ensemble_size'].q
        if isinstance(config_space['dropout'], Constant):
            self.DROPOUT_RANGE = (config_space['dropout'].value, config_space['dropout'].value)
            self.DQ = 1
        else:
            self.DROPOUT_RANGE = (config_space['dropout'].lower, config_space['dropout'].upper)
            self.DQ = config_space['dropout'].q

        self.STACK_UPPER = config_space['stack_layers'].upper
        self.cache = np.full((
            (self.SIZE_RANGE[1]-self.SIZE_RANGE[0]) // self.SQ + 1,
            (self.RATIO_RANGE[1]-self.RATIO_RANGE[0]) // self.RQ + 1,
            (self.DROPOUT_RANGE[1]-self.DROPOUT_RANGE[0]) // self.DQ + 1
        ), False)

        self.iteration_id = 0
        self.incumbent_config = None
        self.incumbent_perf = -np.inf


    def get_history(self) -> History:
        assert self.config_advisor is not None
        return self.config_advisor.history

    def run(self) -> History:
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return np.max(self.perfs)


    def iterate(self, budget=MAX_INT) -> Observation:

        _start_time = time.time()
        # get configuration suggestion from advisor
        i = 0
        max_sample_num = 5000 # self.RATIO_RANGE[1] * self.SIZE_RANGE[1] * (self.DROPOUT_RANGE[1] + 1)
        while True:
            config = self.config_advisor.get_suggestion().get_dictionary().copy()
            if isinstance(self.config_space['dropout'], Constant):
                config['dropout'] = self.config_space['dropout'].value
            size = config['ensemble_size']
            ratio = config['ratio']
            dropout = config['dropout']
            if self.cache[(size-self.SIZE_RANGE[0])//self.SQ, (ratio-self.RATIO_RANGE[0])//self.RQ, (dropout-self.DROPOUT_RANGE[0])//self.DQ]:
                i += 1
                if i > max_sample_num:
                    logger.info("Can't sample new config after %d tries! Randomly choose one!" % max_sample_num)

                    availables = np.where(self.cache == False)
                    idx = np.random.randint(0, len(availables[0]))
                    s1, s2, s3 = availables[0][idx], availables[1][idx], availables[2][idx]
                    self.cache[s1, s2, s3] = True

                    config.update({'ensemble_size': self.SIZE_RANGE[0]+s1*self.SQ, 'ratio': self.RATIO_RANGE[0]+s2*self.RQ, 'dropout': self.DROPOUT_RANGE[0]+s3*self.DQ})
                    break
            else:
                self.cache[(size-self.SIZE_RANGE[0])//self.SQ, (ratio-self.RATIO_RANGE[0])//self.RQ, (dropout-self.DROPOUT_RANGE[0])//self.DQ] = True
                break
        config.update({'stack_layers': self.STACK_UPPER, 'meta_learner': 'auto'})

        timeout = self.per_run_time_limit
        if np.isinf(timeout):
            timeout = None

        # evaluate configuration on objective_function
        obj_args, obj_kwargs = (config,), dict()
        result = run_obj_func(self.evaluator, obj_args, obj_kwargs, timeout)

        # parse result
        ret, timeout_status, traceback_msg, elapsed_time = (
            result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
        if timeout_status:
            trial_state = TIMEOUT
        elif traceback_msg is not None:
            trial_state = FAILED
            logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
        else:
            trial_state = SUCCESS

        for leader, objective in ret['leader_board'].items():
            tmp = leader.split("-")
            head = tmp[0].split('_')[0]
            layer = int(tmp[1].split("L")[1])
            config_dict = config.copy()
            config_dict.update({'meta_learner': head, 'stack_layers': layer-1})
            new_config = get_config_from_dict(self.config_space, config_dict)
            # update observation to advisor
            observation = Observation(
                config=new_config, objectives=[objective],
                trial_state=trial_state, elapsed_time=elapsed_time,
            )
            self.config_advisor.update_observation(observation)
            self.update_saver([new_config], [objective])
            self.configs.append(new_config)
            self.perfs.append(-objective)

        self.incumbent_perf = self.evaluator.best_pool.best_perfs[-1]
        self.incumbent_config = deepcopy(self.evaluator.best_pool.best_configs[-1])

        self.iteration_id += 1
        # Logging
        logger.info('Iter %d, leader_boards: %s.' % (self.iteration_id, str(ret['leader_board'])))

        if np.all(self.cache):
            logger.info('Have already try all base model combinations! Early stop!')
            self.early_stopped_flag = True

        if self.time_limit is not None and time.time() - self.timestamp > self.time_limit or \
                self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit:
            self.timeout_flag = True

        iteration_cost = time.time() - _start_time

        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_opt_trajectory(self):

        trajectory = {
            'detail_perfs': ",".join([str(p) for p in self.perfs]),
        }

        return trajectory

    def get_incumbent_config(self):
        incumbent_config = self.incumbent_config[0].copy()

        return incumbent_config
