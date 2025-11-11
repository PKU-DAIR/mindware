import time
import os
import numpy as np
from openbox.utils.constants import SUCCESS, FAILED

from mindware.components.optimizers.base_optimizer import BaseOptimizer, MAX_INT
from mindware.components.optimizers.base.mfsebase import MfseBase


class MfseOptimizer(BaseOptimizer, MfseBase):
    def __init__(self, evaluator, config_space, name, eval_type, 
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, 
                 inner_iter_num_per_iter=1, timestamp=None,
                 R=27, eta=3, 
                 output_dir='./', seed=1, n_jobs=1, topk=50):
        BaseOptimizer.__init__(evaluator=evaluator, config_space=config_space, name=name, eval_type=eval_type, 
                               time_limit=time_limit, evaluation_limit=evaluation_limit, 
                               per_run_time_limit=per_run_time_limit, per_run_mem_limit=per_run_mem_limit, 
                               inner_iter_num_per_iter=inner_iter_num_per_iter, timestamp=timestamp, 
                               output_dir=output_dir, seed=seed, topk=topk)
        MfseBase.__init__(self, eval_func=self.evaluator, config_space=self.config_space,
                          per_run_time_limit=per_run_time_limit, seed=seed,
                          R=R, eta=eta, n_jobs=n_jobs, output_dir=output_dir)

    def iterate(self, budget=MAX_INT):
        """
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        """
        _start_time = time.time()
        for _ in range(self.inner_iter_num_per_iter):
            _time_elapsed = time.time() - _start_time
            if _time_elapsed >= budget:
                break
            budget_left = budget - _time_elapsed
            iter_full_eval_configs, iter_full_eval_perfs = self._iterate(self.s_values[self.inner_iter_id], budget=budget_left)
            self.update_saver(iter_full_eval_configs, iter_full_eval_perfs)
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

            # Remove tmp model
            if self.evaluator.continue_training:
                for filename in os.listdir(self.evaluator.model_dir):
                    # Temporary model
                    if 'tmp_%s' % self.evaluator.timestamp in filename:
                        try:
                            filepath = os.path.join(self.evaluator.model_dir, filename)
                            os.remove(filepath)
                        except Exception:
                            pass

        if len(self.full_eval_perfs) > 0:
            inc_idx = np.argmin(np.array(self.full_eval_perfs))

            for idx in range(len(self.full_eval_perfs)):
                if self.name in ['hpofe', 'cash', 'cashfe']:
                    if hasattr(self.evaluator, 'fe_config'):
                        fe_config = self.evaluator.fe_config
                    else:
                        fe_config = None
                    self.eval_dict[(fe_config, self.full_eval_configs[idx])] = [-self.full_eval_perfs[idx],
                                                                                time.time(), FAILED if np.isinf(
                            self.full_eval_perfs[idx]) else SUCCESS]
                else:
                    if hasattr(self.evaluator, 'hpo_config'):
                        hpo_config = self.evaluator.hpo_config
                    else:
                        hpo_config = None
                    self.eval_dict[(self.full_eval_configs[idx], hpo_config)] = [-self.full_eval_perfs[idx],
                                                                                 time.time(), FAILED if np.isinf(
                            self.full_eval_perfs[idx]) else SUCCESS]

            self.incumbent_perf = -self.full_eval_perfs[inc_idx]
            self.incumbent_config = self.full_eval_configs[inc_idx]

        self.perfs = [-loss for loss in self.full_eval_perfs]
        self.configs = self.full_eval_configs

        # Incumbent performance: the large, the better.
        iteration_cost = time.time() - _start_time
        
        if self.time_limit is not None and time.time() - self.timestamp > self.time_limit or \
                self.evaluation_num_limit is not None and len(self.perfs) >= self.evaluation_num_limit:
            self.timeout_flag = True
            
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_evaluation_stats(self):
        return self.evaluation_stats
