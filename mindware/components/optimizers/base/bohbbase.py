import time
import numpy as np
import random as rd
from math import log, ceil
from mindware.utils.constant import MAX_INT
from mindware.components.optimizers.base.acquisition import EI
from mindware.components.transfer_learning.tlbo.models.kde import TPE
from mindware.components.optimizers.base.acq_optimizer import RandomSampling
from mindware.components.optimizers.base.funcs import get_types, std_normalization
from mindware.components.optimizers.base.config_space_utils import sample_configurations
from mindware.components.optimizers.base.config_space_utils import convert_configurations_to_array
from mindware.components.computation.parallel_process import ParallelProcessEvaluator
from mindware.utils.logging_utils import get_logger
from mindware.components.optimizers.base.prob_rf import RandomForestWithInstances
from mindware.utils.decorators import time_limit


class BohbBase(object):
    def __init__(self, eval_func, config_space, config_generator='tpe', per_run_time_limit=600,
                 seed=1, R=27, eta=3, n_jobs=1):
        self.eval_func = eval_func
        self.config_space = config_space
        self.config_generator = config_generator
        self.n_workers = n_jobs
        self.per_run_time_limit = per_run_time_limit

        self.trial_cnt = 0
        self.full_eval_configs = list()
        self.full_eval_perfs = list()
        self.global_start_time = time.time()
        self.time_ticks = list()
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        # Parameters in Hyperband framework.
        self.restart_needed = True
        self.R = R
        self.eta = eta
        self.seed = seed
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))  # 3
        self.B = (self.s_max + 1) * self.R  # 4 * 27, 一整个SH 需要的资源数
        self.s_values = list(reversed(range(self.s_max + 1)))  # 3, 2, 1, 0
        self.inner_iter_id = 0

        # Parameters in BOHB.
        self.iterate_r = list()
        self.target_x = dict()
        self.target_y = dict()
        self.exp_output = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):  # 0, 3, 9, 27
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        types, bounds = get_types(self.config_space)
        self.num_config = len(bounds)
        self.surrogate = RandomForestWithInstances(types, bounds)

        # self.executor = ParallelEvaluator(self.eval_func, n_worker=n_jobs)
        # self.executor = ParallelProcessEvaluator(self.eval_func, n_worker=n_jobs)
        self.acquisition_func = EI(model=self.surrogate)
        self.acq_optimizer = RandomSampling(self.acquisition_func,
                                            self.config_space,
                                            n_samples=2000,
                                            rng=np.random.RandomState(seed))

        self.config_gen = TPE(config_space)

    def _iterate(self, s, budget=MAX_INT, skip_last=0):
        # Set initial number of configurations
        n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
        # initial number of iterations per config
        r = int(self.R * self.eta ** (-s))

        # Choose a batch of configurations in different mechanisms.
        start_time = time.time()
        T = self.get_candidate_configurations(n)
        time_elapsed = time.time() - start_time
        self.logger.info("Choosing next batch of configurations took %.2f sec." % time_elapsed)

        iter_full_eval_configs = list()
        iter_full_eval_perfs = list()
        with ParallelProcessEvaluator(self.eval_func, n_worker=self.n_workers) as executor:
            for i in range((s + 1) - int(skip_last)):  # changed from s + 1
                if time.time() >= budget + start_time:
                    break

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_resource = r * self.eta ** i

                resource_ratio = float(n_resource / self.R)

                self.logger.info("BOHB: %d configurations x size %d / %d each" %
                                 (int(n_configs), n_resource, self.R))

                if self.n_workers > 1:
                    val_losses = executor.parallel_execute(T, resource_ratio=resource_ratio,
                                                           eta=self.eta,
                                                           first_iter=(i == 0))

                    for _id, _val_loss in enumerate(val_losses):
                        if isinstance(_val_loss, dict):
                            val_losses[_id] = _val_loss['objectives'][0]

                    for _id, _val_loss in enumerate(val_losses):
                        if np.isfinite(_val_loss):
                            self.target_x[int(n_resource)].append(T[_id])
                            self.target_y[int(n_resource)].append(_val_loss)

                else:
                    val_losses = list()
                    for config in T:
                        if time.time() - start_time > budget:
                            self.logger.warning('Time limit exceeded!')
                            break

                        try:
                            with time_limit(self.per_run_time_limit):
                                val_loss = self.eval_func(config, resource_ratio=resource_ratio,
                                                          eta=self.eta, first_iter=(i == 0))
                        except Exception as e:
                            val_loss = np.inf

                        if isinstance(val_loss, dict):
                            val_loss = val_loss['objectives'][0]

                        val_losses.append(val_loss)

                        if np.isfinite(val_loss):
                            self.target_x[int(n_resource)].append(config)
                            self.target_y[int(n_resource)].append(val_loss)

                self.exp_output[time.time()] = (int(n_resource), T, val_losses)

                self.logger.info('iter: %d, resource: %d, val_loss: %s' % (i, n_resource, str(val_losses)))

                if int(n_resource) == self.R:
                    self.full_eval_configs.extend(T)
                    self.full_eval_perfs.extend(val_losses)
                    iter_full_eval_configs.extend(T)
                    iter_full_eval_perfs.extend(val_losses)
                    self.time_ticks.extend([time.time() - self.global_start_time] * len(T))

                    # Only update results using maximal resources
                    if self.config_generator != 'smac':
                        for _id, _val_loss in enumerate(val_losses):
                            if np.isfinite(_val_loss):
                                self.config_gen.new_result(T[_id], _val_loss)

                # Select a number of best configurations for the next loop.
                # Filter out early stops, if any.
                indices = np.argsort(val_losses)
                if len(T) >= self.eta:
                    T = [T[i] for i in indices]
                    reduced_num = int(n_configs / self.eta)
                    T = T[0:reduced_num]
                else:
                    T = [T[indices[0]]]

        # Refit the surrogate model.
        resource_val = self.iterate_r[-1]
        if len(self.target_y[resource_val]) > 1:
            if self.config_generator == 'smac':
                normalized_y = std_normalization(self.target_y[resource_val])
                self.surrogate.train(convert_configurations_to_array(self.target_x[resource_val]),
                                     np.array(normalized_y, dtype=np.float64))
                
        return iter_full_eval_configs, iter_full_eval_perfs

    def smac_get_candidate_configurations(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) <= 3:
            return sample_configurations(self.config_space, num_config, logger=self.logger)

        incumbent = dict()
        max_r = self.iterate_r[-1]
        best_index = np.argmin(self.target_y[max_r])
        incumbent['config'] = self.target_x[max_r][best_index]
        incumbent['obj'] = self.target_y[max_r][best_index]
        self.acquisition_func.update(model=self.surrogate, eta=incumbent)

        config_candidates = self.acq_optimizer.maximize(batch_size=num_config)
        p_threshold = 0.3
        candidates = list()
        idx_acq = 0
        for _id in range(num_config):
            if rd.random() < p_threshold or _id >= len(config_candidates):
                i = 0
                config = sample_configurations(self.config_space, 1)[0]
                while config in candidates:
                    config = sample_configurations(self.config_space, 1)[0]
                    i += 1
                    if i > 1000:
                        self.logger.warning('Cannot sample a new random configuration after 1000 iters.')
                        break
            else:
                config = config_candidates[idx_acq]
                idx_acq += 1
            candidates.append(config)
        return candidates

    def baseline_get_candidate_configurations(self, num_config):
        config_candidates = list()

        config_left = num_config
        while config_left:
            i = 0
            config = self.config_gen.get_config()[0]
            while config in config_candidates:
                config = self.config_gen.get_config()[0]
                i += 1
                if i > 1000:
                    self.logger.warning('Cannot sample a new configuration after 1000 iters.')
                    break
            config_candidates.append(config)
            config_left -= 1

        p_threshold = 0.3
        candidates = list()
        idx_acq = 0
        for _id in range(num_config):
            if rd.random() < p_threshold:
                i = 0
                config = sample_configurations(self.config_space, 1)[0]
                while config in candidates:
                    config = sample_configurations(self.config_space, 1)[0]
                    i += 1
                    if i > 1000:
                        self.logger.warning('Cannot sample a new random configuration after 1000 iters.')
                        break
            else:
                config = config_candidates[idx_acq]
                idx_acq += 1
            if config not in config_candidates:
                candidates.append(config)

        return candidates

    def get_candidate_configurations(self, num_config):
        if self.config_generator == 'smac':
            return self.smac_get_candidate_configurations(num_config)
        else:
            return self.baseline_get_candidate_configurations(num_config)
