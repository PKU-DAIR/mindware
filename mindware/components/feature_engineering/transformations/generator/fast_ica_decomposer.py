import warnings
from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, NotEqualsCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from mindware.components.utils.configspace_utils import check_for_bool, check_none
import sklearn

class FastIcaDecomposer(Transformer):
    type = 10

    def __init__(self, algorithm='parallel', whiten='False', fun='logcosh', n_components=None,
                 random_state=1):
        super().__init__("fast_ica")
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components

        self.random_state = random_state
        self.skip_flag = False
        self.pre_trained = False

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X, y = input_datanode.data

        # Skip heavy computation in fast ica.
        if X.shape[0] > 10000 or X.shape[1] > 200:
            if not self.pre_trained:
                pass
                # self.skip_flag = True
        self.pre_trained = True
        if self.skip_flag:
            return X.copy()

        if self.model is None:
            from sklearn.decomposition import FastICA

            if sklearn.__version__ <= '1.0.2':
                self.whiten = check_for_bool(self.whiten)
            else:
                if self.whiten == 'False':
                    self.whiten = False

            if check_none(self.n_components):
                self.n_components = None
            else:
                self.n_components = int(self.n_components)

            if self.n_components is not None:
                self.n_components = min(self.n_components, X.shape[0])

            self.model = FastICA(
                n_components=self.n_components, algorithm=self.algorithm,
                fun=self.fun, whiten=self.whiten, random_state=self.random_state
            )
            # Make the RuntimeWarning an Exception!
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message='array must not contain infs or NaNs')
                try:
                    self.model.fit(X)
                except ValueError as e:
                    if 'array must not contain infs or NaNs' in e.args[0]:
                        raise ValueError("Bug in scikit-learn: https://github.com/scikit-learn/scikit-learn/pull/2738")
                    raise e

        X_new = self.model.transform(X)
        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        meta_mask = kwargs.get('meta', False)

        # n_components cannot be larger than min(n_features, n_samples).
        n_samples = kwargs.get('n_samples', None)
        n_features = kwargs.get('n_features', None)
        exp_deflation_mask = kwargs.get('exp_deflation_mask', True) | meta_mask
        cube_parallel_mask = kwargs.get('cube_parallel_mask', True) | meta_mask

        
        n_components_lower = 1
        n_components_upper = 2000
        if not meta_mask:
            n_components_lower = 10
            if n_samples is not None:
                n_components_upper = min(n_components_upper, n_samples)
            if n_features is not None:
                n_components_upper = min(n_components_upper, n_features)
            
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            if n_components_upper <= 10:
                n_components = Constant("n_components", n_components_upper)
            else:
                n_components = UniformIntegerHyperparameter(
                    "n_components", n_components_lower, n_components_upper, default_value=min(100, n_components_upper))
            algorithm = CategoricalHyperparameter('algorithm',
                                                  ['parallel', 'deflation'], 'parallel')
            if sklearn.__version__ <= '1.0.2':
                whiten = CategoricalHyperparameter('whiten',
                                                   ['False', 'True'], 'False')
            else:
                whiten = CategoricalHyperparameter('whiten',
                                                   ['False', 'unit-variance', 'arbitrary-variance'], 'unit-variance')

            fun = CategoricalHyperparameter(
                'fun', ['logcosh', 'exp', 'cube'], 'logcosh')
            cs.add_hyperparameters([n_components, algorithm, whiten, fun])
            cs.add_condition(NotEqualsCondition(n_components, whiten, "False"))
            if not exp_deflation_mask:
                fun_and_algorithm = ForbiddenAndConjunction(
                    ForbiddenEqualsClause(whiten, "False"),
                    ForbiddenEqualsClause(fun, "exp"),
                    ForbiddenEqualsClause(algorithm, "deflation"))
                cs.add_forbidden_clause(fun_and_algorithm)
                
            if not cube_parallel_mask:
                fun_and_algorithm = ForbiddenAndConjunction(
                    ForbiddenEqualsClause(whiten, "False"),
                    ForbiddenEqualsClause(fun, "cube"),
                    ForbiddenEqualsClause(algorithm, "parallel"))
                cs.add_forbidden_clause(fun_and_algorithm)
            
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_components': hp.randint('ica_n_components', 1990) + 10,
                     'algorithm': hp.choice('ica_algorithm', ['parallel', 'deflation']),
                     'whiten': 'False',
                     'fun': hp.choice('ica_fun', ['logcosh', 'exo', 'cube'])}
            return space
