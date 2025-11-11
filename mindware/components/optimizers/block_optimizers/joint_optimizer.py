import time
import numpy as np
from copy import deepcopy

from mindware.components.optimizers.base_optimizer import BaseOptimizer
from ConfigSpace import ConfigurationSpace, Constant
from mindware.components.optimizers.smac_optimizer import SMACOptimizer
from mindware.components.optimizers.random_search_optimizer import RandomSearchOptimizer
from mindware.components.optimizers.tpe_optimizer import TPEOptimizer
from mindware.components.optimizers.bohb_optimizer import BohbOptimizer
from mindware.components.optimizers.mfse_optimizer import MfseOptimizer
from mindware.utils.constant import MAX_INT


class JointOptimizer(BaseOptimizer):
    def __init__(self, node_list, node_index,
                 evaluator, cash_config_space, name, eval_type,
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024,
                 inner_iter_num_per_iter=10, timestamp=None,
                 sub_optimizer='smac', fe_config_space_dict=None,
                 output_dir='./', seed=1, n_jobs=1, topk=50):
        super(JointOptimizer, self).__init__(evaluator=evaluator, config_space=(cash_config_space, fe_config_space_dict), name=name, eval_type=eval_type, 
                                             time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                             per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                             inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                             output_dir=output_dir, seed=seed, topk=topk)
        self.eval_dict = dict()

        # TODO: Support asynchronous BO
        if sub_optimizer == 'random_search':
            optimizer_class = RandomSearchOptimizer
        elif sub_optimizer == 'tpe':
            optimizer_class = TPEOptimizer
        elif sub_optimizer == 'smac':
            optimizer_class = SMACOptimizer
        elif sub_optimizer == 'bohb':
            optimizer_class = BohbOptimizer
        elif sub_optimizer == 'mfse':
            optimizer_class = MfseOptimizer
        else:
            raise ValueError("Invalid optimizer %s" % sub_optimizer)

        assert cash_config_space is not None or fe_config_space_dict is not None

        cs = ConfigurationSpace()
        if cash_config_space is not None:
            cs.add_hyperparameters(
                deepcopy(cash_config_space.get_hyperparameters()))
            cs.add_conditions(
                deepcopy(cash_config_space.get_conditions()))
            cs.add_forbidden_clauses(
                deepcopy(cash_config_space.get_forbiddens()))
        if fe_config_space_dict is not None:
            for _fe_config_space in fe_config_space_dict.values():
                for hp in _fe_config_space.get_hyperparameters():
                    try:
                        cs.add_hyperparameter(deepcopy(hp))
                    except:
                        pass
                for cd in _fe_config_space.get_conditions():
                    try:
                        cs.add_condition(deepcopy(cd))
                    except:
                        pass
                for fb in _fe_config_space.get_forbiddens():
                    try:
                        cs.add_forbidden_clause(deepcopy(fb))
                    except:
                        pass

        self.sub_bandit = optimizer_class(
            evaluator=self.evaluator, config_space=cs, name='hpofe',
            eval_type=self.eval_type, output_dir=self.output_dir,
            time_limit=time_limit, evaluation_limit=None,
            per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit,
            inner_iter_num_per_iter=inner_iter_num_per_iter,
            timestamp=self.timestamp, seed=self.seed, n_jobs=n_jobs, topk=topk
        )

        if self.time_limit is not None:
            self.evaluation_num_limit = MAX_INT

        self.timeout_flag = False

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return self.incumbent_perf

    def iterate(self, budget=MAX_INT):
        _start_time = time.time()

        self.sub_bandit.inner_iter_num_per_iter = self.inner_iter_num_per_iter
        self.sub_bandit.iterate(budget=budget)

        self.incumbent_perf = self.sub_bandit.incumbent_perf
        self.incumbent_config = self.sub_bandit.incumbent_config
        self.perfs = self.sub_bandit.perfs
        self.configs = self.sub_bandit.configs

        # Update stop flag
        self.early_stopped_flag = self.sub_bandit.early_stopped_flag
        self.timeout_flag = self.sub_bandit.timeout_flag
        if self.early_stopped_flag:
            self.logger.info("Maximum configuration number met for each arm candidate!")
        if self.timeout_flag:
            self.logger.info('Time elapsed!')

        iteration_cost = time.time() - _start_time

        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_opt_trajectory(self):

        trajectory = {
            'detail_perfs': ",".join([str(p) for p in self.perfs])
        }

        return trajectory
