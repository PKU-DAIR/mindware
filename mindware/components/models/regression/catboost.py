import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from mindware.components.models.base_model import BaseRegressionModel
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class CatboostRegressor(BaseRegressionModel):
    def __init__(self, iterations, learning_rate, depth, l2_leaf_reg, loss_function,
                 random_state=None, embed_min_categories=2):
        BaseRegressionModel.__init__(self)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_state = random_state
        self.estimator = None
        self.embed_min_categories = embed_min_categories

    def recover_X(self, X):
        X_output = X.copy()
        # 转化为pandas.DataFrame
        if not isinstance(X_output, pd.DataFrame):
            X_output = pd.DataFrame(X_output)

        feature_map = self.feature_map

        map_dict = dict()
        for i, key in enumerate(feature_map):
            if key not in map_dict:
                map_dict[key] = [i]
            else:
                map_dict[key].append(i)

        need_pro = []

        for key, inds in map_dict.items():
            if len(inds) >= self.embed_min_categories:
                need_pro.append(inds)

        start_n = len(feature_map) - sum([len(inds) for inds in need_pro])
        ind_number_map = dict()
        delete_inds = []

        for i,inds in enumerate(need_pro):
            ind_number_map[start_n+i] = len(inds)
            new_X = X_output.iloc[:, inds].idxmax(axis=1).astype(str)
            new_column = f"recover_{start_n+i}"  # Rename the column to avoid duplication

            X_output[new_column] = new_X
            delete_inds += inds

        X_output.drop(X_output.columns[delete_inds], axis=1, inplace=True)
        self.ind_number_map = ind_number_map
        return X_output

    def fit(self, X, Y, sample_weight=None):
        from catboost import CatBoostRegressor

        X = self.recover_X(X)

        self.iterations = int(self.iterations)
        self.learning_rate = float(self.learning_rate)
        self.depth = int(self.depth)
        self.l2_leaf_reg = float(self.l2_leaf_reg)

        estimator = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_state,
            verbose=False,
            cat_features=list(self.ind_number_map.keys())
        )

        estimator.fit(X, Y, sample_weight=sample_weight)
        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError

        X = self.recover_X(X)
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CatBoostReg',
                'name': 'CatBoost Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        cs = ConfigurationSpace()

        iterations = UniformIntegerHyperparameter(
            name="iterations", lower=50, upper=1000, default_value=200, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=0.5, default_value=0.1, log=True)
        depth = UniformIntegerHyperparameter(
            name="depth", lower=2, upper=10, default_value=6, log=False)
        l2_leaf_reg = UniformFloatHyperparameter(
            name="l2_leaf_reg", lower=1, upper=10, default_value=3, log=False)
        loss_function = CategoricalHyperparameter(
            name="loss_function",
            choices=["RMSE", "MAE"],
            default_value="RMSE"
        )

        cs.add_hyperparameters([iterations, learning_rate, depth, l2_leaf_reg, loss_function])
        return cs
