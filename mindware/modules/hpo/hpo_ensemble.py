import os
import pickle as pkl

from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.classification import _classifiers, _addons as _cls_addons
from mindware.components.models.regression import _regressors, _addons as _rgs_addons

from mindware.modules.hpo.hpo_evaluator import get_hpo_cs
from mindware.modules.hpo.base_hpo import BaseHPOOptimizer

from mindware.components.ensemble.ensemble_bulider import EnsembleBuilder


class EnsembleHPOOptimizer(BaseHPOOptimizer):
    def __init__(self, estimator_id: str, task_type=None, scorer: str = 'acc',
                 data_node: DataNode = None, evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', per_run_time_limit=600,
                 time_limit=600, amount_of_resource=None,
                 output_dir=None, seed=None, n_jobs=1,
                 ensemble_method='ensemble_selection', ensemble_size=5):

        super().__init__(estimator_id=estimator_id, task_type=task_type, scorer=scorer,
                         data_node=data_node, evaluation=evaluation, resampling_params=resampling_params,
                         optimizer=optimizer, per_run_time_limit=per_run_time_limit,
                         time_limit=time_limit, amount_of_resource=amount_of_resource,
                         output_dir=output_dir, seed=seed, n_jobs=n_jobs)

        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-HPO-Ensemble-%s-%s(%d)' % (self.estimator_id, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def run(self):

        final_ensemble = False

        for i in range(self.amount_of_resource):
            if not (self.early_stop_flag or self.timeout_flag):
                self.iterate()

        self.ensemble_res = -self.evaluator.evaluate_ensemble(self.ensemble_method, self.ensemble_size)['objectives'][0]
        print('Final ensemble score is %s' % str(self.ensemble_res))

        return self.incumbent_perf
