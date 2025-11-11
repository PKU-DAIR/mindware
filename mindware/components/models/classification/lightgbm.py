# 修改后
import numpy as np
import pandas as pd
from mindware.components.models.base_model import BaseClassificationModel
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, CategoricalHyperparameter, UnParametrizedHyperparameter

class LightGBM(BaseClassificationModel):
    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, colsample_bytree, augment_data=0, random_state=None, verbose=-1):
        BaseClassificationModel.__init__(self)
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.num_leaves = int(num_leaves)
        self.max_depth = int(max_depth)
        self.subsample = subsample
        self.min_child_samples = int(min_child_samples)
        self.colsample_bytree = colsample_bytree
        self.augment_data = augment_data
        
        # self.n_estimators = 120
        # self.learning_rate = 4e-2
        # self.num_leaves = 31
        # self.max_depth = 15
        # self.subsample = 0.85
        # self.min_child_samples = 1000
        # self.colsample_bytree = 1.0
        # self.augment_data = 0
        
        
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

        self.var_counts = {}
        self.var_neg_corr = {}
        self.features = None
        self.n_jobs = 1
        

    def fit(self, X, Y):
        from lightgbm import LGBMClassifier
        from sklearn.utils.multiclass import unique_labels
        self.classes_ = unique_labels(Y)

        if self.augment_data == 1:
            print("Augmenting data...")
            self.target = Y
            X = X.values if isinstance(X, pd.DataFrame) else X
            X, Y = self.augment_data_func(X, Y)
            print(f"Shape after augment_data_func - X: {X.shape}, y: {len(Y)}")

        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs,
                                        verbose=self.verbose)

        if self.augment_data == 1:
            self.estimator.fit(X, Y,
                               categorical_feature=[2])
        else:
            self.estimator.fit(X, Y)

        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.augment_data == 1:
            y_pred = self.predict_for_each_feature(X, mode="predict")
        else:
            y_pred = self.estimator.predict(X)

        return y_pred

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.augment_data == 1:
            y_pred = self.predict_for_each_feature(X, mode="predict_proba")
        else:
            y_pred = self.estimator.predict_proba(X)
        return y_pred

    def predict_for_each_feature(self, X, mode="predict_proba"):
        if isinstance(X, pd.DataFrame):
            X = X.values

        y_preds = []

        for idx in range(X.shape[1]):

            tmp = self.var_to_feat(
                feature_data=X[:, idx],
                feature_id=idx,
                is_train=False
            )

            y_pred = self.estimator.predict_proba(tmp)[:, 1]
            y_preds.append(y_pred)

        y_preds = np.array(y_preds)
        y_preds = np.sum(self.logit(y_preds), axis=0)

        if mode == "predict":
            y_preds = (y_preds > 0).astype(int)
            print(np.sum(y_preds))
        else:
            from scipy.stats import rankdata
            y_preds = self.sigmoid(y_preds)
            y_preds = rankdata(y_preds) / len(y_preds)
            y_preds = np.vstack([1 - y_preds, y_preds]).T

        return y_preds

        #     y_pred = self.estimator.predict_proba(tmp)
        #     y_preds.append(y_pred)

        # y_preds = np.array(y_preds)
        # y_preds = np.sum(np.log(y_preds + 1e-15), axis=0)
        # final_y_preds = np.zeros(y_preds.shape)
        # for i in range(y_preds.shape[1]):
        #     tmp = y_preds - y_preds[:,i:i+1]
        #     final_y_preds[:, i] = 1 / np.exp(tmp).sum(axis=1)

        # if mode == "predict":
        #     final_y_preds = np.argmax(final_y_preds, axis=1)
        # else:
        #     from scipy.stats import rankdata
        #     for i in range(final_y_preds.shape[1]):
        #         final_y_preds[:,i] = rankdata(-final_y_preds[:,i]) / len(final_y_preds[:,i])

        # return final_y_preds

    def augment_data_func(self, X, y):
        X_df = pd.DataFrame(X)
        features = X_df.columns

        print(f"Running augment_data_func, X_df shape: {X_df.shape}, y shape: {len(y)}")

        augmented_data = []
        augmented_labels = []

        for idx, feature in enumerate(features):
            # print(f"Processing feature index: {idx}")

            feature_data = X_df[feature]

            if idx not in self.var_counts:
                self.var_counts[idx] = feature_data.value_counts()

            tmp = self.var_to_feat(
                feature_data=feature_data,
                feature_id=idx,
                is_train=True
            )

            augmented_data.append(tmp)
            augmented_labels.append(y)

        X_augmented = np.vstack(augmented_data)
        y_augmented = np.concatenate(augmented_labels)

        print(f"Final augmented data shape - X: {X_augmented.shape}, y: {len(y_augmented)}")
        return X_augmented, y_augmented

    def var_to_feat(self, feature_data, feature_id, is_train=False):
        new_df = pd.DataFrame(columns=["var", "count", "feature_id", "var_rank"])
        new_df["var"] = feature_data.values if isinstance(feature_data, pd.Series) else feature_data

        var_stats = self.var_counts[feature_id]

        new_df["count"] = pd.Series(feature_data).map(var_stats)
        new_df["count"] = new_df["count"].fillna(0)
        new_df["feature_id"] = feature_id
        new_df["var_rank"] = new_df["var"].rank(method='average') / len(new_df)

        if is_train and hasattr(self, 'target'):
            corr = np.corrcoef(self.target, new_df["var"])[0, 1]
            if corr < 0:
                # print(f"Feature {feature_id} is negatively correlated with target. Multiplying by -1.")
                new_df["var"] = -new_df["var"]
                self.var_neg_corr[feature_id] = True
            else:
                self.var_neg_corr[feature_id] = False
        else:
            if self.var_neg_corr.get(feature_id, False):
                # print(f"Feature {feature_id} is negatively correlated with target. Multiplying by -1.")
                new_df["var"] = -new_df["var"]

        return new_df.values

    def logit(self, p):
        return np.log(p + 1e-15) - np.log(1 - p + 1e-15)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LightGBM Classifier',
                'name': 'LightGBM Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'input': ('SPARSE', 'DENSE', 'UNSIGNED_DATA'),
                'output': ('PREDICTIONS',)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        n_classes = kwargs.get('n_classes', None)
        
        aug_choices = [0]
        # if n_classes is not None and n_classes == 2:
        #     aug_choices = [0, 1]
        # augment_data = CategoricalHyperparameter("augment_data", aug_choices, default_value=0)
        
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
        max_depth = UniformIntegerHyperparameter('max_depth', 3, 15, default_value=15)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
        min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 1000, default_value=20)
        subsample = UniformFloatHyperparameter("subsample", 0.7, 1.0, default_value=1.0, q=0.1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1.0, default_value=1.0, q=0.1)
        # verbose = UnParametrizedHyperparameter("verbose", -1)
        cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                                colsample_bytree])
        return cs
