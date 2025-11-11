from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, CategoricalHyperparameter
import numpy as np
import pandas as pd
from mindware.components.utils.constants import *
from mindware.components.models.base_model import BaseRegressionModel

class LightGBM(BaseRegressionModel):
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda, augment_data=0,
                 random_state=1, verbose=-1):
        BaseRegressionModel.__init__(self)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.augment_data = augment_data
        self.n_jobs = 1
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.var_mean = {}  # 保存特征的均值
        self.var_var = {}   # 保存特征的方差
        self.features = None

    def fit(self, X, y):
        from lightgbm import LGBMRegressor

        # print(f"Initial shape of X: {X.shape}, y: {len(y)}")

        if self.augment_data == 1:
            print("Augmenting data...")
            X, y = self.augment_data_func(X, y)
            print(f"Shape after augment_data_func - X: {X.shape}, y: {len(y)}")
            print(f"Data types after augmentation: {pd.DataFrame(X).dtypes}")

        self.features = X.shape[1]  # 保存特征数量
        # print(f"Training LightGBM Regressor model with {self.features} features")

        self.estimator = LGBMRegressor(num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators,
                                       min_child_weight=self.min_child_weight,
                                       subsample=self.subsample,
                                       colsample_bytree=self.colsample_bytree,
                                       reg_alpha=self.reg_alpha,
                                       reg_lambda=self.reg_lambda,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs,
                                       verbose=self.verbose)

        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.augment_data == 1:
            # 对每个特征扩展进行预测
            y_pred = self.predict_for_each_feature(X)
        else:
            y_pred = self.estimator.predict(X)

        return y_pred

    def predict_for_each_feature(self, X):
        features = X.shape[1]
        y_preds = []

        # 确保 X 是 numpy array
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        for i in range(features):
            X_single_feature = X[:, i].reshape(-1, 1)  # 选择单个特征
            X_augmented, _ = self.augment_data_func(X_single_feature, np.zeros(X_single_feature.shape[0]))

            # 对扩展后的特征进行预测
            y_pred_augmented = self.estimator.predict(X_augmented)

            # 将每次预测结果扩展到原始数据的大小
            y_preds.append(y_pred_augmented)

        # 对不同特征的预测结果进行平均
        y_preds = np.vstack(y_preds).mean(axis=0)

        return y_preds

    # 扩展数据的方法，完整迁移
    def augment_data_func(self, X, y):
        X_df = pd.DataFrame(X)
        features = X_df.columns

        print(f"Running augment_data_func, X_df shape: {X_df.shape}, y shape: {len(y)}")

        augmented_data = []
        for feature in features:
            print(f"Processing feature: {feature} - type: {X_df[feature].dtype}")

            if np.issubdtype(X_df[feature].dtype, np.number):
                tmp = self.augment_numeric_feature(X_df[feature], feature)
            else:
                tmp = self.augment_categorical_feature(X_df[feature], feature)

            print(f"Shape of augmented data for {feature}: {tmp.shape}")
            augmented_data.append(tmp)

        X_augmented = np.vstack(augmented_data)
        y_augmented = np.tile(y, len(features))  # 扩展标签为特征数量的倍数

        print(f"Final augmented data shape - X: {X_augmented.shape}, y: {len(y_augmented)}")
        return X_augmented, y_augmented

    def augment_numeric_feature(self, feature_data, feature_name):
        # 对数值型特征进行归一化，并保存均值和方差
        print(f"Augmenting numeric feature: {feature_name}")
        if feature_name not in self.var_mean:
            self.var_mean[feature_name] = feature_data.mean()
            self.var_var[feature_name] = feature_data.var()

        normalized_feature = (feature_data - self.var_mean[feature_name]) / self.var_var[feature_name]
        tmp = self.var_to_feat(normalized_feature)
        return tmp

    def augment_categorical_feature(self, feature_data, feature_name):
        # 对类别型特征进行频率编码
        print(f"Augmenting categorical feature: {feature_name}")
        freq_map = feature_data.value_counts() / len(feature_data)
        feature_data_encoded = feature_data.map(freq_map).astype(float)  # 这里确保编码后是浮点型

        tmp = self.var_to_feat(feature_data_encoded)
        return tmp

    def var_to_feat(self, vr):
        new_df = pd.DataFrame()
        new_df["var"] = vr.values
        new_df["var_rank"] = new_df["var"].rank() / len(vr)  # 归一化排名

        return new_df.values

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LightGBM Regressor',
                'name': 'LightGBM Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 1023, default_value=31)
            learning_rate = UniformFloatHyperparameter("learning_rate", 0.025, 0.3, default_value=0.1, log=True)
            min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
            subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1)
            reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-10, 10, log=True, default_value=1e-10)
            reg_lambda = UniformFloatHyperparameter("reg_lambda", 1e-10, 10, log=True, default_value=1e-10)
            # augment_data = CategoricalHyperparameter("augment_data", [0], default_value=0)  # 控制扩展数据的选项
            # verbose = UnParametrizedHyperparameter("verbose", -1)
            cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_weight, subsample,
                                    colsample_bytree, reg_alpha, reg_lambda])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': hp.randint('lgb_n_estimators', 901) + 100,
                     'num_leaves': hp.randint('lgb_num_leaves', 993) + 31,
                     'learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.025), np.log(0.3)),
                     'min_child_weight': hp.randint('lgb_min_child_weight', 10) + 1,
                     'subsample': hp.uniform('lgb_subsample', 0.5, 1),
                     'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.5, 1),
                     'reg_alpha': hp.loguniform('lgb_reg_alpha', np.log(1e-10), np.log(10)),
                     'reg_lambda': hp.loguniform('lgb_reg_lambda', np.log(1e-10), np.log(10))
                     }

            init_trial = {'n_estimators': 500,
                          'num_leaves': 31,
                          'learning_rate': 0.1,
                          'min_child_weight': 1,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'reg_alpha': 1e-10,
                          'reg_lambda': 1e-10
                          }

            return space
