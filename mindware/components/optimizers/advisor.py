# License: MIT

import os
import abc
import numpy as np
from datetime import datetime

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler, HaltonSampler
from openbox.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
import openbox
from openbox.core.generic_advisor import Advisor
from ConfigSpace import ConfigurationSpace, Constant
from copy import deepcopy



class MyAdvisor(Advisor):
    """
    Basic Advisor Class, which adopts a policy to sample a configuration.
    """

    sub_areas = ['ensemble_size', 'ratio', 'dropout']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def alter_model(self, history: History):

        num_config_evaluated = len(history)

        if num_config_evaluated >= 400:
            if self.surrogate_type == 'gp':
                self.surrogate_type = 'prf'
                logger.info(f'n_observations={num_config_evaluated}, change surrogate model from GP to PRF!')
                self.setup_bo_basics()

    def sample_random_configs(self, num_configs=1, history=None, excluded_configs=None):
        """
        Sample a batch of random configurations.
        Parameters
        ----------
        num_configs

        history

        Returns
        -------

        """
        if history is None:
            history = self.history
        if excluded_configs is None:
            excluded_configs = set()

        keys = sorted(list(self.sub_cs.keys()))
        records = set()
        for config in history.configurations + list(excluded_configs):
            key = '-'.join([str(config[x]) for x in keys])
            records.add(key)

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.sub_cs.sample_configuration()
            sample_cnt += 1
            # if config not in (history.configurations + configs) and config not in excluded_configs:
            key = '-'.join([str(config[x]) for x in keys])
            if key not in records:
                records.add(key)
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs


    def create_initial_design(self, init_strategy='default'):
        """
        Create several configurations as initial design.
        Parameters
        ----------
        init_strategy: str

        Returns
        -------
        Initial configurations.
        """

        if not hasattr(self, 'sub_cs'):
            self.sub_cs = ConfigurationSpace()
            for hyper in self.config_space.get_hyperparameters():
                if hyper.name not in MyAdvisor.sub_areas or isinstance(hyper, Constant): continue
                self.sub_cs.add_hyperparameter(deepcopy(hyper))
            self.sub_cs.seed(self.config_space_seed * 2)

        default_config = self.config_space.get_default_configuration()
        num_random_config = self.init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_random_configs(self.init_num)
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_random_configs(num_random_config, excluded_configs=[default_config])
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_random_configs(100)
            initial_configs = self.max_min_distance(default_config, candidate_configs, num_random_config)
        elif init_strategy == 'sobol':
            sobol = SobolSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + sobol.generate(return_config=True)
        elif init_strategy == 'latin_hypercube':
            lhs = LatinHypercubeSampler(self.sub_cs, num_random_config, criterion='maximin', random_state=self.config_space_seed)
            initial_configs = [default_config] + lhs.generate(return_config=True)
        elif init_strategy == 'halton':
            halton = HaltonSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + halton.generate(return_config=True)
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

        valid_configs = []
        for config in initial_configs:
            try:
                config.is_valid_configuration()
            except ValueError:
                continue
            valid_configs.append(config)
        if len(valid_configs) != len(initial_configs):
            logger.warning('Only %d/%d valid configurations are generated for initial design strategy: %s. '
                                'Add more random configurations.'
                                % (len(valid_configs), len(initial_configs), init_strategy))
            num_random_config = self.init_num - len(valid_configs)
            valid_configs += self.sample_random_configs(num_random_config, excluded_configs=valid_configs)
        return valid_configs

    def max_min_distance(self, default_config, src_configs, num, sel_idx=[0, 1, 3]):
        min_dis = list()
        initial_configs = list()
        initial_configs.append(default_config)

        for config in src_configs:
            dis = np.linalg.norm(config.get_array()[sel_idx] - default_config.get_array()[sel_idx])
            min_dis.append(dis)
        min_dis = np.array(min_dis)

        for i in range(num):
            furthest_config = src_configs[np.argmax(min_dis)]
            initial_configs.append(furthest_config)
            min_dis[np.argmax(min_dis)] = -1

            for j in range(len(src_configs)):
                if src_configs[j] in initial_configs:
                    continue
                updated_dis = np.linalg.norm(src_configs[j].get_array()[sel_idx] - furthest_config.get_array()[sel_idx])
                min_dis[j] = min(updated_dis, min_dis[j])

        return initial_configs

    def get_suggestion(self, history: History = None, return_list: bool = False):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history is None:
            history = self.history

        self.alter_model(history)

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if len(self.initial_configurations) > 0:
            res = self.initial_configurations[0]
            self.initial_configurations = self.initial_configurations[1:]
            return [res] if return_list else res
        if self.optimization_strategy == 'random':
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        if (not return_list) and self.rng.random() < self.rand_prob:
            logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')
        cY = history.get_constraints(transform='bilog')

        if self.optimization_strategy == 'bo':
            if num_config_successful < max(self.init_num, 1):
                logger.warning('No enough successful initial trials! Sample random configuration.')
                res = self.sample_random_configs(1, history)[0]
                return [res] if return_list else res

            # train surrogate model
            if self.num_objectives == 1:
                self.surrogate_model.train(X, Y[:, 0])
            elif self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objectives)
                weights = weights / np.sum(weights)
                scalarized_obj = get_chebyshev_scalarization(weights, Y)
                self.surrogate_model.train(X, scalarized_obj(Y))
            else:  # multi-objectives
                for i in range(self.num_objectives):
                    self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
            for i in range(self.num_constraints):
                self.constraint_models[i].train(X, cY[:, i])

            # update acquisition function
            if self.num_objectives == 1:
                incumbent_value = history.get_incumbent_value()
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated)
            else:  # multi-objectives
                mo_incumbent_values = history.get_mo_incumbent_values()
                if self.acq_type == 'parego':
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     eta=scalarized_obj(np.atleast_2d(mo_incumbent_values)),
                                                     num_data=num_config_evaluated)
                elif self.acq_type.startswith('ehvi'):
                    partitioning = NondominatedPartitioning(self.num_objectives, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1])
                else:
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     constraint_perfs=cY,  # for MESMOC
                                                     eta=mo_incumbent_values,
                                                     num_data=num_config_evaluated,
                                                     X=X, Y=Y)

            # optimize acquisition function
            if openbox.__version__ <= '0.8.1':
                challengers = self.optimizer.maximize(runhistory=history,
                                                    num_points=5000).challengers
            else:
                challengers =  self.acq_optimizer.maximize(acquisition_function=self.acquisition_function,
                                                        history=history,
                                                        num_points=5000)
            if return_list:
                # Caution: return_list doesn't contain random configs sampled according to rand_prob
                return challengers

            keys = sorted(list(self.sub_cs.keys()))
            records = set()
            for config in history.configurations:
                key = '-'.join([str(config[x]) for x in keys])
                records.add(key)
            for config in challengers:
                # if config not in history.configurations:
                #     return config
                key = '-'.join([str(config[x]) for x in keys])
                if key not in records:
                    return config
            logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                'Sample random config.' % (len(challengers), ))
            return self.sample_random_configs(1, history)[0]
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)
