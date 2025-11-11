import numpy as np
import pandas as pd
from mindware.components.models.base_model import BaseRegressionModel

def get_estimator(name):
    if name == "NN_TORCH":
        from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
        return TabularNeuralNetTorchModel
    elif name == "GBM":
        from autogluon.tabular.models.lgb.lgb_model import LGBModel
        return LGBModel

    elif name == "CAT":
        from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
        return CatBoostModel
    
    elif name == "XGB":
        from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
        return XGBoostModel
    elif name == "FASTAI":
        from autogluon.tabular.models.fastainn.tabular_nn_fastai import NNFastAiTabularModel
        return NNFastAiTabularModel
    elif name == "RF":
        from autogluon.tabular.models.rf.rf_model import RFModel
        return RFModel
    elif name == "XT":
        from autogluon.tabular.models.xt.xt_model import XTModel
        return XTModel
    elif name == "KNN":
        from autogluon.tabular.models.knn.knn_model import KNNModel
        return KNNModel



class Autogluon_wraper(BaseRegressionModel):
    def __init__(self,
                 model_name,
                 params,
                 eval_metric,
                 problem_type,
                 random_state,
                 ):
        BaseRegressionModel.__init__(self)
        self.model_name = model_name
        self.params = params

        self.estimator = get_estimator(self.model_name)(hyperparameters = params, eval_metric = eval_metric, problem_type=problem_type)

    
    def fit(self, X, Y, X_val=None, y_val=None):
        self.estimator.fit(X =X, y=Y, X_val =X_val, y_val = y_val)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)


    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Autogluon wraper',
                'name': 'Autogluon wraper',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'input': ('SPARSE', 'DENSE', 'UNSIGNED_DATA'),
                'output': ('PREDICTIONS',)}


    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        return None