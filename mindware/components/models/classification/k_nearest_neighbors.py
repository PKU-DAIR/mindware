from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class KNearestNeighborsClassifier(BaseClassificationModel):

    def __init__(self, n_neighbors, weights, p, random_state=None):
        BaseClassificationModel.__init__(self)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state
        self.n_jobs = 1
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.multiclass
        from sklearn.utils.multiclass import unique_labels
        self.classes_ = unique_labels(Y)

        estimator = \
            sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                                   weights=self.weights,
                                                   p=self.p, n_jobs=self.n_jobs)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KNN',
                'name': 'K-Nearest Neighbor Classification',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        meta_mask = kwargs.get('meta', False)

        # Expected n_neighbors <= n_samples
        n_samples = kwargs.get('n_samples', None)
        n_neighbors_upper = 100
        if not meta_mask:
            n_neighbors_upper = min(100, n_samples) if n_samples is not None else 100

        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_neighbors = UniformIntegerHyperparameter(
                name="n_neighbors", lower=1, upper=n_neighbors_upper, log=True, default_value=1)
            weights = CategoricalHyperparameter(
                name="weights", choices=["uniform", "distance"], default_value="uniform")
            p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
            cs.add_hyperparameters([n_neighbors, weights, p])

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_neighbors': hp.randint('knn_n_neighbors', n_neighbors_upper) + 1,
                     'weights': hp.choice('knn_weights', ['uniform', 'distance']),
                     'p': hp.choice('knn_p', [1, 2])}

            init_trial = {'n_neighbors': 1, 'weights': "uniform", 'p': 2}

            return space
