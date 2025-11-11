import abc
import os
import time
import datetime
import numpy as np
from mindware.utils.constant import MAX_INT
from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from ConfigSpace.configuration_space import ConfigurationSpace


class BaseOptimizer(object):
    def __init__(self, evaluator: _BaseEvaluator, config_space, name, eval_type, 
                 time_limit=None, evaluation_limit=None,
                 per_run_time_limit=300, per_run_mem_limit=1024, 
                 inner_iter_num_per_iter=1, timestamp=None, 
                 output_dir='./', seed=None, topk=50):
        self.evaluator = evaluator
        self.config_space = config_space

        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.inner_iter_num_per_iter = inner_iter_num_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit

        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        if isinstance(self.config_space, ConfigurationSpace):
            self.incumbent_config = self.config_space.get_default_configuration().get_dictionary()
        elif isinstance(self.config_space, tuple):
            tmp = {}
            for cs in self.config_space:
                if isinstance(cs, ConfigurationSpace):
                    tmp.update(cs.get_default_configuration().get_dictionary().copy())
                elif isinstance(cs, dict):
                    tmp.update(cs[list(cs.keys())[0]].get_default_configuration().get_dictionary().copy())
            self.incumbent_config = tmp
        self.eval_dict = dict()

        assert name in ['hpo', 'hpofe', 'fe', 'cash', 'cashfe', 'ens']
        self.name = name
        self.seed = np.random.random_integers(MAX_INT) if seed is None else seed
        self.start_time = time.time()
        self.timing_list = list()
        self.eval_type = eval_type
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.init_hpo_iter_num = None
        self.early_stopped_flag = False
        self.timeout_flag = False
        self.timestamp = timestamp
        if self.timestamp is None:
            self.timestamp = time.time()
        self.output_dir = output_dir
        self.topk_saver = CombinedTopKModelSaver(
            k=topk, model_dir=self.output_dir,
            identifier=datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        )

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def iterate(self, budget=MAX_INT):
        pass

    # TODO：Refactor the other optimizers
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
                if self.eval_type in ['holdout', 'partial', 'partial_bohb', 'cv']:
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

    def get_evaluation_stats(self):
        return

    def gc(self):
        return

    def get_opt_trajectory(self):
        return None
