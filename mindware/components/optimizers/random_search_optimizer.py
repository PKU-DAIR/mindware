import time
import numpy as np
from openbox.optimizer.generic_smbo import SMBO as RandomSearch
from openbox.optimizer.parallel_smbo import pSMBO as pRandomSearch

from mindware.components.utils.constants import SUCCESS
from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT


class RandomSearchOptimizer(BaseOptimizer):

    def __init__(self, evaluator, config_space, name, eval_type, 
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, 
                 inner_iter_num_per_iter=1, timestamp=None,
                 output_dir='./', seed=1, n_jobs=1, topk=50):
        super(RandomSearchOptimizer, self).__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type, 
                                                    time_limit=time_limit, evaluation_limit=evaluation_limit, 
                                                    per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                                                    inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                                                    output_dir=output_dir, seed=seed, topk=topk)

        if n_jobs == 1:
            self.optimizer = RandomSearch(objective_function=self.evaluator,
                                          config_space=config_space,
                                          advisor_type='random',
                                          task_id='Default',
                                          max_runtime_per_trial=self.per_run_time_limit,
                                          random_state=self.seed)
        else:
            self.optimizer = pRandomSearch(objective_function=self.evaluator,
                                           config_space=config_space,
                                           sample_strategy='random',
                                           batch_size=n_jobs,
                                           task_id='Default',
                                           max_runtime_per_trial=self.per_run_time_limit,
                                           random_state=self.seed)

        self.trial_cnt = 0
        self.exp_output = dict()

        hp_num = len(self.config_space.get_hyperparameters())
        if hp_num == 0:
            self.config_num_threshold = 0
        else:
            _threshold = int(len(set(self.config_space.sample_configuration(5000))))
            self.config_num_threshold = _threshold

        self.logger.debug("The maximum trial number in HPO is :%d" % self.config_num_threshold)
        self.maximum_config_num = min(1500, self.config_num_threshold)
        self.n_jobs = n_jobs

    def run(self):
        while True:
            if self.early_stopped_flag or self.timeout_flag:
                break
            self.iterate()
        return np.max(self.perfs)

    def iterate(self, budget=MAX_INT):
        _start_time = time.time()

        if len(self.configs) == 0 and self.init_hpo_iter_num is not None:
            inner_iter_num = self.init_hpo_iter_num
            print('initial hpo trial num is set to %d' % inner_iter_num)
        else:
            inner_iter_num = self.inner_iter_num_per_iter

        if self.n_jobs == 1:
            for _ in range(inner_iter_num):
                if len(self.configs) >= self.maximum_config_num:
                    self.early_stopped_flag = True
                    self.logger.warning('Already explored 70 percentage of the '
                                        'hyperspace or maximum configuration number met: %d!' % self.maximum_config_num)
                    break
                if time.time() - _start_time > budget:
                    self.logger.warning('Time limit exceeded!')
                    break
                
                obs = self.optimizer.iterate()
                _config, _status, _perf = obs.config, obs.trial_state, obs.objectives
                self.update_saver([_config], [_perf[0]])
                if _status == SUCCESS:
                    self.exp_output[time.time()] = (_config, _perf[0])
                    self.configs.append(_config)
                    self.perfs.append(-_perf[0])
        else:
            if len(self.configs) >= self.maximum_config_num:
                self.early_stopped_flag = True
                self.logger.warning('Already explored 70 percentage of the '
                                    'hyperspace or maximum configuration number met: %d!' % self.maximum_config_num)
            elif time.time() - _start_time > budget:
                self.logger.warning('Time limit exceeded!')
            else:
                
                obs_list = self.optimizer.async_iterate(n=inner_iter_num)
                _config_list, _perf_list = [], []
                for obs in obs_list:
                    _config, _perf = obs.config, obs.objectives
                    _config_list.append(_config)
                    _perf_list.append(_perf[0])
                    if obs.trial_state == SUCCESS:
                        self.exp_output[time.time()] = (_config, _perf[0])
                        self.configs.append(_config)
                        self.perfs.append(-_perf[0])
                        
                self.update_saver(_config_list, _perf_list)
                
        run_history = self.optimizer.get_history()
        if self.name in ['hpofe', 'cash', 'cashfe']:
            if hasattr(self.evaluator, 'fe_config'):
                fe_config = self.evaluator.fe_config
            else:
                fe_config = None
        
            self.eval_dict.update({
                (fe_config, run_history.configurations[i]): [-run_history.objectives[i][0], time.time(), run_history.trial_states[i]]
                for i in range(max(-len(run_history), -self.inner_iter_num_per_iter), 0)
            })
        else:
            if hasattr(self.evaluator, 'hpo_config'):
                hpo_config = self.evaluator.hpo_config
            else:
                hpo_config = None
            self.eval_dict.update({
                (run_history.configurations[i], hpo_config): [-run_history.objectives[i][0], time.time(), run_history.trial_states[i]]
                for i in range(max(-len(run_history), -self.inner_iter_num_per_iter), 0)
            })
            
        if len(run_history.get_incumbents()) > 0:
            incumbent = run_history.get_incumbents()[0]
            self.incumbent_config, self.incumbent_perf = incumbent.config.get_dictionary().copy(), incumbent.objectives[0]
            self.incumbent_perf = -self.incumbent_perf
        iteration_cost = time.time() - _start_time
        
        if self.time_limit is not None and time.time() - self.timestamp > self.time_limit or \
                self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit:
            self.timeout_flag = True

        return self.incumbent_perf, iteration_cost, self.incumbent_config
