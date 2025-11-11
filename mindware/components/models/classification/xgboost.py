import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class XGBoostClassifier(BaseClassificationModel):

    def __init__(self, n_estimators, learning_rate, max_depth, min_child_weight,
                 subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, random_state=None):
        BaseClassificationModel.__init__(self)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_child_weight = int(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.gamma = float(gamma)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.random_state = random_state
        self.n_jobs = 1
        self.estimator = None
        '''
            def __init__(self, n_estimators, learning_rate, max_depth, min_child_weight,
                        subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, random_state=None):
                self.n_estimators = n_estimators
                self.learning_rate = learning_rate
                self.max_depth = max_depth
                self.min_child_weight = min_child_weight
                self.subsample = subsample
                self.colsample_bytree = colsample_bytree
                self.gamma = gamma
                self.reg_alpha = reg_alpha
                self.reg_lambda = reg_lambda
                self.random_state = None
                self.estimator = None
        '''
        
    def fit(self, X, Y, sample_weight=None):
        import xgboost as xgb
        from sklearn.utils.multiclass import unique_labels
        self.classes_ = unique_labels(Y)
        
        estimator = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        estimator.fit(X, Y, sample_weight=sample_weight)
        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'XGB',
                'name': 'XGBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=100, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=0.5, default_value=0.1, log=True)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=3, upper=10, default_value=6, log=False)
        min_child_weight = UniformIntegerHyperparameter(
            name="min_child_weight", lower=1, upper=10, default_value=1, log=False)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.5, upper=1, default_value=1, log=False)
        colsample_bytree = UniformFloatHyperparameter(
            name="colsample_bytree", lower=0.5, upper=1, default_value=1, log=False)
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=0, upper=1, default_value=0, log=False)
        reg_alpha = UniformFloatHyperparameter(
            name="reg_alpha", lower=0, upper=10, default_value=0, log=False)
        reg_lambda = UniformFloatHyperparameter(
            name="reg_lambda", lower=1, upper=10, default_value=1, log=False)

        cs.add_hyperparameters([n_estimators, learning_rate, max_depth, min_child_weight,
                                subsample, colsample_bytree, gamma, reg_alpha, reg_lambda])
        return cs
