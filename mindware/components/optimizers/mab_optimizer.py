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


class MabOptimizer(BaseOptimizer):
    def __init__(self, evaluator, config_space, name, eval_type,
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, 
                 inner_iter_num_per_iter=10, timestamp=None,
                 output_dir='./', seed=1, n_jobs=1, topk=50,
                 sub_optimizer='smac', fe_config_space=None):
        super(MabOptimizer, self).__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type,
                                           time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                           per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                           inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                           output_dir=output_dir, seed=seed, topk=topk)

        # Bandit settings.
        self.alpha = 4
        self.arms = list(config_space.get_hyperparameter('algorithm').choices)
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()

        self.arm_cost_stats = dict()
        for _arm in self.arms:
            self.arm_cost_stats[_arm] = list()

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

        for arm in self.arms:

            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()

            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', arm))
            # Add active hyperparameters
            hps = config_space.get_hyperparameters()
            for hp in hps:
                if hp.name.split(':')[0] == arm:
                    cs.add_hyperparameter(hp)
            # Add active conditions
            conds = config_space.get_conditions()
            for cond in conds:
                try:
                    cs.add_condition(cond)
                except:
                    pass
            # Add active forbidden clauses
            forbids = config_space.get_forbiddens()
            for forbid in forbids:
                try:
                    cs.add_forbidden_clause(forbid)
                except:
                    pass
            if fe_config_space is not None:
                cs.add_hyperparameters(
                    deepcopy(fe_config_space.get_hyperparameters()))
                cs.add_conditions(
                    deepcopy(fe_config_space.get_conditions()))
                cs.add_forbidden_clauses(
                    deepcopy(fe_config_space.get_forbiddens()))

            self.sub_bandits[arm] = optimizer_class(
                self.evaluator, cs, 'hpo',
                eval_type=self.eval_type, output_dir=self.output_dir,
                time_limit=time_limit, evaluation_limit=None,
                per_run_time_limit=per_run_time_limit,
                inner_iter_num_per_iter=self.inner_iter_num_per_iter,
                timestamp=self.timestamp, seed=self.seed, n_jobs=n_jobs
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.time_records = list()

        # Initialize the parameters.
        self.pull_cnt = 0
        self.pick_id = 0
        self.update_cnt = 0
        arm_num = len(self.arms)
        self.optimal_algo_id = None
        self.arm_candidate = self.arms.copy()
        self.best_lower_bounds = np.zeros(arm_num)

        if self.time_limit is None:
            if arm_num * self.alpha > self.evaluation_num_limit:
                raise ValueError('Trial number should be larger than %d.' % (arm_num * self.alpha))
        else:
            self.evaluation_num_limit = MAX_INT

        self.timeout_flag = False

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return self.incumbent_perf

    def iterate(self, budget=MAX_INT):

        for _arm in self.arms:
            self.sub_bandits[_arm].inner_iter_num_per_iter = self.inner_iter_num_per_iter

        _start_time = time.time()
        # Search for an arm that is not early-stopped.
        while self.sub_bandits[self.arm_candidate[self.pick_id]].early_stopped_flag and \
                self.pick_id < len(self.arm_candidate):
            self.pick_id += 1

        if self.pick_id < len(self.arm_candidate):
            # Pull the arm.
            arm_to_pull = self.arm_candidate[self.pick_id]
            self.logger.info('Optimize %s in the %d-th iteration' % (arm_to_pull, self.pull_cnt))
            _start_time = time.time()
            self.sub_bandits[arm_to_pull].inner_iter_num_per_iter = self.inner_iter_num_per_iter
            reward, _, incumbent = self.sub_bandits[arm_to_pull].iterate(budget=self.time_limit + self.timestamp - time.time())

            self.perfs.extend(self.sub_bandits[arm_to_pull].perfs[-self.inner_iter_num_per_iter:])
            self.configs.extend(self.sub_bandits[arm_to_pull].configs[-self.inner_iter_num_per_iter:])

            # Update results after each iteration
            self.arm_cost_stats[arm_to_pull].append(time.time() - _start_time)
            if reward > self.incumbent_perf:
                self.optimal_algo_id = arm_to_pull
                self.incumbent_perf = reward
                self.incumbent_config = incumbent
            self.eval_dict.update(self.sub_bandits[arm_to_pull].eval_dict)
            self.rewards[arm_to_pull].append(reward)
            self.action_sequence.append(arm_to_pull)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.timestamp)
            # self.logger.info('The best performance found for %s is %.4f' % (arm_to_pull, reward))
            self.pull_cnt += 1
            self.pick_id += 1

            # Logger output
            scores = list()
            for _arm in self.arms:
                scores.append(self.sub_bandits[_arm].incumbent_perf)
            scores = np.array(scores)
            self.logger.info('=' * 50)
            self.logger.info('Best_algo_perf:  %s' % str(self.incumbent_perf))
            self.logger.info('Best_algo_id:    %s' % str(self.optimal_algo_id))
            self.logger.info('Arm candidates:  %s' % str(self.arms))
            self.logger.info('Best val scores: %s' % str(list(scores)))
            self.logger.info('=' * 50)

        # Eliminate arms after pulling each arm a few times.
        if self.pick_id == len(self.arm_candidate):
            self.update_cnt += 1
            self.pick_id = 0
            # Update the arms until pulling each arm for at least alpha times.
            if self.update_cnt >= self.alpha:
                # Update the upper/lower bound estimation.
                budget_left = max(self.time_limit - time.time() + self.timestamp, 0)
                avg_cost = np.array([np.mean(self.arm_cost_stats[_arm]) for _arm in self.arm_candidate]).mean()
                steps = int(budget_left / avg_cost)
                upper_bounds, lower_bounds = list(), list()

                for _arm in self.arm_candidate:
                    rewards = self.rewards[_arm]
                    slope = (rewards[-1] - rewards[-self.alpha]) / self.alpha
                    if self.time_limit is None:
                        steps = self.evaluation_num_limit - self.pull_cnt
                    upper_bound = np.min([1.0, rewards[-1] + slope * steps])
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(rewards[-1])
                    self.best_lower_bounds[self.arms.index(_arm)] = rewards[-1]

                # Reject the sub-optimal arms.
                n = len(self.arm_candidate)
                flags = [False] * n
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if upper_bounds[i] < lower_bounds[j]:
                                flags[i] = True
                for i in range(n):
                    if np.isnan(upper_bounds[i]) or not np.isfinite(lower_bounds[i]):
                        flags[i] = True

                if np.sum(flags) == n:
                    self.logger.error('Removing all the arms simultaneously!')

                self.logger.info('=' * 50)
                self.logger.info('Candidates  : %s' % ','.join(self.arm_candidate))
                self.logger.info('Upper bound : %s' % ','.join(['%.4f' % val for val in upper_bounds]))
                self.logger.info('Lower bound : %s' % ','.join(['%.4f' % val for val in lower_bounds]))
                self.logger.info(
                    'Arms removed: %s' % [item for idx, item in enumerate(self.arm_candidate) if flags[idx]])
                self.logger.info('=' * 50)

                # Update arm_candidates.
                self.arm_candidate = [item for index, item in enumerate(self.arm_candidate) if not flags[index]]

        # Update stop flag
        self.early_stopped_flag = True
        self.timeout_flag = False
        for _arm in self.arm_candidate:
            if not self.sub_bandits[_arm].early_stopped_flag:
                self.early_stopped_flag = False
        if self.early_stopped_flag:
            self.logger.info(
                "Maximum configuration number met for each arm candidate!")
        if time.time() - self.timestamp > self.time_limit or self.pull_cnt >= self.evaluation_num_limit:
            self.timeout_flag = True
            self.logger.info('Time elapsed!')

        iteration_cost = time.time() - _start_time

        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_opt_trajectory(self):

        trajectory = {
            'action_sequence': self.action_sequence,
            'rewards_of_bandits': self.rewards,
            'final_rewards': self.final_rewards,
            'detail_perfs': ",".join([str(p) for p in self.perfs])
        }

        return trajectory


