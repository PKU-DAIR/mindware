import time
import numpy as np
from copy import deepcopy

from mindware.components.optimizers.base_optimizer import BaseOptimizer
from ConfigSpace import ConfigurationSpace, Constant
from mindware.utils.constant import MAX_INT


class ConditionalOptimizer(BaseOptimizer):
    def __init__(self, node_list, node_index,
                 evaluator, cash_config_space, name, eval_type,
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024,
                 inner_iter_num_per_iter=10, timestamp=None,
                 sub_optimizer='smac', fe_config_space_dict=None,
                 output_dir='./', seed=1, n_jobs=1, topk=50):

        super(ConditionalOptimizer, self).__init__(evaluator=evaluator, config_space=(cash_config_space, fe_config_space_dict), name=name, eval_type=eval_type, 
                                                   time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                                   per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                                   inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                                   output_dir=output_dir, seed=seed, topk=topk)

        self.node_index = node_index

        # Bandit settings.
        self.alpha = 4
        self.arms = list(cash_config_space.get_hyperparameter('algorithm').choices)
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()

        self.arm_cost_stats = dict()
        for _arm in self.arms:
            self.arm_cost_stats[_arm] = list()

        for arm in self.arms:

            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()

            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', arm))
            # Add active hyperparameters
            hps = cash_config_space.get_hyperparameters()
            for hp in hps:
                if hp.name.split(':')[0] == arm:
                    cs.add_hyperparameter(hp)
            # Add active conditions
            conds = cash_config_space.get_conditions()
            for cond in conds:
                try:
                    cs.add_condition(cond)
                except:
                    pass
            # Add active forbidden clauses
            forbids = cash_config_space.get_forbiddens()
            for forbid in forbids:
                try:
                    cs.add_forbidden_clause(forbid)
                except:
                    pass

            from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_node_type
            child_type = get_opt_node_type(node_list, node_index + 1)
            self.sub_bandits[arm] = child_type(
                node_list=node_list, node_index=node_index + 1,
                evaluator=self.evaluator, cash_config_space=deepcopy(cs), name='hpofe', eval_type=self.eval_type, 
                time_limit=time_limit, evaluation_limit=None,
                per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp, 
                sub_optimizer=sub_optimizer, fe_config_space_dict={arm: fe_config_space_dict[arm]},
                output_dir=self.output_dir,seed=self.seed, n_jobs=n_jobs, topk=topk
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
            reward, _, incumbent = self.sub_bandits[arm_to_pull].iterate(budget=budget)

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
            self.logger.info('Node index: %s' % str(self.node_index))
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
            if self.sub_bandits[_arm].timeout_flag:
                self.timeout_flag = True
        if self.early_stopped_flag:
            self.logger.info("Maximum configuration number met for each arm candidate!")
        if self.timeout_flag:
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


