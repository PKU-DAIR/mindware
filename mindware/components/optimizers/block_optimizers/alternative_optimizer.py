import time
import numpy as np
from copy import deepcopy
from copy import copy
from ConfigSpace import Configuration, ConfigurationSpace

from mindware.components.optimizers.base_optimizer import BaseOptimizer
from openbox.utils.constants import SUCCESS, TIMEOUT, FAILED
from mindware.utils.constant import MAX_INT


class AlternativeOptimizer(BaseOptimizer):
    def __init__(self, node_list, node_index,
                 evaluator, cash_config_space, name, eval_type,
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024,
                 inner_iter_num_per_iter=10, timestamp=None,
                 sub_optimizer='smac', fe_config_space_dict=None,
                 output_dir='./', seed=1, n_jobs=1, topk=50):

        super(AlternativeOptimizer, self).__init__(evaluator=evaluator,
                                                   config_space=(cash_config_space, fe_config_space_dict), name=name,
                                                   eval_type=eval_type,
                                                   time_limit=time_limit, evaluation_limit=evaluation_limit,
                                                   per_run_time_limit=per_run_time_limit,
                                                   per_run_mem_limit=per_run_mem_limit,
                                                   inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp,
                                                   output_dir=output_dir, seed=seed, topk=topk)

        assert cash_config_space is not None
        assert fe_config_space_dict is not None

        self.mode = 0

        self.node_list = node_list
        self.node_index = node_index
        self.sub_optimizer = sub_optimizer
        self.n_jobs = n_jobs

        self.arms = ['hpo', 'fe']
        self.optimal_algo_id = None
        self.first_start = True
        self.sub_bandits = dict()
        self.rewards = dict()
        self.evaluation_cost = dict()
        self.update_flag = dict()

        self.fe_config_space = ConfigurationSpace()
        for _fe_config_space in fe_config_space_dict.values():
            for hp in _fe_config_space.get_hyperparameters():
                try:
                    self.fe_config_space.add_hyperparameter(deepcopy(hp))
                except:
                    pass
            for cd in _fe_config_space.get_conditions():
                try:
                    self.fe_config_space.add_condition(deepcopy(cd))
                except:
                    pass
            for fb in _fe_config_space.get_forbiddens():
                try:
                    self.fe_config_space.add_forbidden_clause(deepcopy(fb))
                except:
                    pass
                
        # Global incumbent.
        self.init_config = {'fe': self.fe_config_space.get_default_configuration().get_dictionary().copy(),
                            'hpo': cash_config_space.get_default_configuration().get_dictionary().copy()}
        self.inc = {'fe': self.fe_config_space.get_default_configuration().get_dictionary().copy(),
                    'hpo': cash_config_space.get_default_configuration().get_dictionary().copy()}
        self.local_inc = {'fe': self.fe_config_space.get_default_configuration().get_dictionary().copy(),
                          'hpo': cash_config_space.get_default_configuration().get_dictionary().copy()}
        self.local_hist = {'fe': [], 'hpo': []}
        self.inc_record = {'fe': list(), 'hpo': list()}
        self.exp_output = dict()
        self.eval_dict = dict()
        self.arm_eval_dict = {'fe': dict(), 'hpo': dict()}
        for arm in self.arms:
            self.rewards[arm] = list()
            self.update_flag[arm] = False
            self.evaluation_cost[arm] = list()
            self.exp_output[arm] = dict()
        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()

        for arm in self.arms:

            if arm == 'hpo':
                evaluator = copy(self.evaluator)
                evaluator.fixed_config = self.init_config['fe']

                from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_node_type
                child_type = get_opt_node_type(node_list, node_index + 2)
                self.sub_bandits[arm] = child_type(
                    node_list=node_list, node_index=node_index + 2,
                    evaluator=evaluator, cash_config_space=cash_config_space, name='hpo', eval_type=self.eval_type,
                    time_limit=time_limit, evaluation_limit=None,
                    per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit,
                    inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
                    sub_optimizer=sub_optimizer, fe_config_space_dict=None,
                    output_dir=self.output_dir, seed=self.seed, n_jobs=n_jobs, topk=topk
                )
            elif arm == 'fe':
                evaluator = copy(self.evaluator)
                evaluator.fixed_config = self.init_config['hpo']

                from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_node_type
                child_type = get_opt_node_type(node_list, node_index + 1)
                self.sub_bandits[arm] = child_type(
                    node_list=node_list, node_index=node_index + 1,
                    evaluator=evaluator, cash_config_space=None, name='fe', eval_type=self.eval_type,
                    time_limit=time_limit, evaluation_limit=None,
                    per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit,
                    inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
                    sub_optimizer=sub_optimizer, fe_config_space_dict=fe_config_space_dict,
                    output_dir=self.output_dir, seed=self.seed, n_jobs=n_jobs,
                )
            else:
                raise ValueError("Wrong arm name %s." % arm)

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

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return self.incumbent_perf

    def iterate(self, budget=MAX_INT):

        for _arm in self.arms:
            self.sub_bandits[_arm].inner_iter_num_per_iter = self.inner_iter_num_per_iter

        arm_to_pull = self.arms[self.pull_cnt % 2]
        if self.sub_bandits[arm_to_pull].early_stopped_flag:
            arm_to_pull = self.arms[(self.pull_cnt + 1) % 2]

        # if self.first_start and arm_to_pull == 'hpo':
        #     self.sub_bandits['hpo'].inner_iter_num_per_iter = self.inner_iter_num_per_iter * 2
        #     self.first_start = False
        # if self.first_start and arm_to_pull == 'fe':
        #     self.sub_bandits['fe'].inner_iter_num_per_iter = self.inner_iter_num_per_iter * 5
        #     self.first_start = False

        _start_time = time.time()

        reward, _, _ = self.sub_bandits[arm_to_pull].iterate(budget=budget)
        iter_cost = time.time() - _start_time
        self.action_sequence.append(arm_to_pull)
        self.pull_cnt += 1

        # Update results after each iteration
        pre_inc_perf = self.incumbent_perf
        for arm_id in self.arms:
            self.update_flag[arm_id] = False
        self.arm_eval_dict[arm_to_pull].update(self.sub_bandits[arm_to_pull].eval_dict)
        self.eval_dict.update(self.sub_bandits[arm_to_pull].eval_dict)
        self.rewards[arm_to_pull].append(reward)
        self.evaluation_cost[arm_to_pull].append(iter_cost)
        pre_local_inc = self.local_inc[arm_to_pull].copy()
        self.local_inc[arm_to_pull] = self.sub_bandits[arm_to_pull].incumbent_config

        # Update global incumbent from FE and HPO.
        if np.isfinite(reward) and reward > self.incumbent_perf:
            cur_inc = self.sub_bandits[arm_to_pull].incumbent_config
            self.inc[arm_to_pull] = cur_inc
            self.local_hist[arm_to_pull].append(cur_inc)
            self.optimal_algo_id = arm_to_pull
            self.incumbent_perf = reward

            # Alter-HPO strategy: HPO changes if FE changes, FE keeps though HPO changes
            if arm_to_pull == 'fe':
                self.inc['hpo'] = self.init_config['hpo']
            _incumbent = dict()
            _incumbent.update(self.inc['fe'])
            _incumbent.update(self.inc['hpo'])
            self.incumbent_config = _incumbent.copy()

            arm_id = 'fe' if arm_to_pull == 'hpo' else 'hpo'
            if arm_to_pull == 'fe':
                self.reinitialize(arm_id)
            else:
                # Only reinitialize fe blocks once.
                if len(self.rewards[arm_to_pull]) == 1:
                    self.reinitialize(arm_id)
                    if cur_inc != self.init_config['hpo']:
                        self.logger.info('Initial hp_config for FE has changed!')
                    self.init_config['hpo'] = cur_inc

            if self.mode == 0:
                # Evaluate joint result here
                # Alter-HPO specific
                if arm_to_pull == 'fe' and self.sub_bandits['fe'].evaluator.fixed_config != self.local_inc['hpo']:
                    self.logger.info("Evaluate joint performance in node %s" % self.node_index)
                    self.evaluate_joint_perf()

        if self.mode == 1:
            # Evaluate joint result here
            # Alter-HPO specific
            if arm_to_pull == 'fe' and self.local_inc[arm_to_pull] != pre_local_inc and self.sub_bandits['fe'].evaluator.fixed_config != self.local_inc['hpo']:
                self.logger.info("Evaluate joint performance in node %s" % self.node_index)
                self.evaluate_joint_perf()

        # Logger output
        scores = list()
        for _arm in self.arms:
            scores.append(self.sub_bandits[_arm].incumbent_perf)
        scores = np.array(scores)
        self.logger.info('=' * 50)
        self.logger.info('Node index: %s' % str(self.node_index))
        self.logger.info('Best_part_perf: %s' % str(self.incumbent_perf))
        self.logger.info('Best_part: %s' % str(self.optimal_algo_id))
        self.logger.info('Best val scores: %s' % str(list(scores)))
        self.logger.info('=' * 50)

        self.final_rewards.append(self.incumbent_perf)
        post_inc_perf = self.incumbent_perf
        if np.isfinite(pre_inc_perf) and np.isfinite(post_inc_perf):
            self.inc_record[arm_to_pull].append(post_inc_perf - pre_inc_perf)
        else:
            self.inc_record[arm_to_pull].append(0.)

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

    def reinitialize(self, arm_id):
        if arm_id == 'hpo':
            # Build the Feature Engineering component.
            evaluator = copy(self.evaluator)
            evaluator.fixed_config = self.inc['fe'].copy()

            from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_node_type
            child_type = get_opt_node_type(self.node_list, self.node_index + 2)
            self.sub_bandits[arm_id] = child_type(
                node_list=self.node_list, node_index=self.node_index + 2,
                evaluator=evaluator, cash_config_space=self.config_space[0], name='hpo', eval_type=self.eval_type,
                time_limit=self.time_limit, evaluation_limit=None,
                per_run_time_limit=self.per_run_time_limit, per_run_mem_limit=self.per_run_mem_limit,
                inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
                sub_optimizer=self.sub_optimizer, fe_config_space_dict=None,
                output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs)
        elif arm_id == 'fe':
            evaluator = copy(self.evaluator)
            evaluator.fixed_config = self.inc['hpo'].copy()

            from mindware.components.optimizers.block_optimizers.block_opt_utils import get_opt_node_type
            child_type = get_opt_node_type(self.node_list, self.node_index + 1)
            self.sub_bandits[arm_id] = child_type(
                node_list=self.node_list, node_index=self.node_index + 1,
                evaluator=evaluator, cash_config_space=None, name='fe', eval_type=self.eval_type,
                time_limit=self.time_limit, evaluation_limit=None,
                per_run_time_limit=self.per_run_time_limit, per_run_mem_limit=self.per_run_mem_limit,
                inner_iter_num_per_iter=self.inner_iter_num_per_iter, timestamp=self.timestamp,
                sub_optimizer=self.sub_optimizer, fe_config_space_dict=self.config_space[1],
                output_dir=self.output_dir, seed=self.seed, n_jobs=self.n_jobs)

        else:
            raise ValueError("Wrong arm name %s." % arm_id)

        self.logger.info('=' * 30)
        self.logger.info('UPDATE OPTIMIZER: %s' % arm_id)
        self.logger.info('=' * 30)

    # TODO: Need refactoring
    def evaluate_joint_perf(self):

        evaluator = copy(self.evaluator)
        evaluator.fixed_config = self.local_inc['fe'].copy()
        _perf = - evaluator(self.local_inc['hpo'].copy())['objectives'][0]

        loc_inc_fe = Configuration(self.fe_config_space, self.local_inc['fe'].copy())
        loc_inc_hpo = Configuration(self.config_space[0], self.local_inc['hpo'].copy())

        if _perf is not None and np.isfinite(_perf):
            _config = self.local_inc['fe'].copy()
            _config.update(self.local_inc['hpo'].copy())

            # -perf: The larger, the better.
            self.update_saver([_config], [-_perf])

            self.eval_dict[(loc_inc_fe, loc_inc_hpo)] = [_perf,
                                                         time.time(),
                                                         SUCCESS]
        else:
            self.eval_dict[(loc_inc_fe, loc_inc_hpo)] = [_perf,
                                                         time.time(),
                                                         FAILED]
        # Update INC.
        if _perf is not None and np.isfinite(_perf) and _perf > self.incumbent_perf:
            self.inc['hpo'] = self.local_inc['hpo']
            self.inc['fe'] = self.local_inc['fe']
            self.incumbent_perf = _perf
            _incumbent = dict()
            _incumbent.update(self.inc['fe'])
            _incumbent.update(self.inc['hpo'])
            self.incumbent_config = _incumbent.copy()

    def get_opt_trajectory(self):

        trajectory = {
            'action_sequence': self.action_sequence,
            'rewards_of_bandits': self.rewards,
            'final_rewards': self.final_rewards,
            'detail_perfs': ",".join([str(p) for p in self.perfs])
        }

        return trajectory
