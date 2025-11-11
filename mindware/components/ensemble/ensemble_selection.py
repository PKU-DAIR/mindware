from collections import Counter
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics._scorer import _BaseScorer, _PredictScorer, _ThresholdScorer

from mindware.components.utils.constants import *
from mindware.components.ensemble.base_ensemble import BaseEnsembleModel
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.modules.base_evaluator import fetch_predict_results
from mindware.components.utils.topk_saver import CombinedTopKModelSaver


class EnsembleSelection(BaseEnsembleModel):
    def __init__(
            self, stats,
            ensemble_size: int,
            task_type: int, if_imbal: bool,
            metric: _BaseScorer, resampling_params = None,
            output_dir=None, seed=None,
            sorted_initialization: bool = False,
            mode: str = 'fast',
            predictions=None,
    ):
        super().__init__(stats,
                        ensemble_method='ensemble_selection',
                        ensemble_size=ensemble_size,
                        task_type=task_type, if_imbal=if_imbal,
                        metric=metric, resampling_params=resampling_params,
                        output_dir=output_dir, seed=seed,
                        predictions=predictions)
        self.base_model_mask = None
        self.sorted_initialization = sorted_initialization
        self.mode = mode
        self.encoder = OneHotEncoder()
        self.random_state = np.random.RandomState(1)
        self.shape = self.predictions[0].shape

    def calculate_score(self, pred, y_true):
        if isinstance(self.metric, _ThresholdScorer):
            if len(y_true.shape) == 1:
                y_true = self.encoder.transform(np.reshape(y_true, (len(y_true), 1))).toarray()
        elif self.task_type in CLS_TASKS and isinstance(self.metric, _PredictScorer):
            pred = np.argmax(pred, axis=-1)
        score = -np.inf
        try:
            score = self.metric._score_func(y_true, pred) * self.metric._sign
        except:
            self.logger.info("Error when cal score!")
        return score

    def equal_fit(self):
        self.num_input_models_ = len(self.predictions)
        assert self.num_input_models_ == self.ensemble_size
        weights = np.full((self.num_input_models_,), 1 / self.num_input_models_)

        self.weights_ = weights
        self.base_model_mask = np.full(len(self.predictions), False)
        self.base_model_mask[self.weights_ != 0] = True

    def fit(self, datanode):
        if isinstance(datanode, DataNode):
            train_labels = datanode.data[1]
        elif isinstance(datanode, np.ndarray):
            train_labels = datanode
        elif isinstance(datanode, list):
            train_labels = np.array(datanode)
        else:
            raise ValueError("Wrong training node type:%s" % str(type(datanode)))

        if len(train_labels.shape) == 1 and self.task_type in CLS_TASKS:
            reshape_y = np.reshape(train_labels, (len(train_labels), 1))
            self.encoder.fit(reshape_y)
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not self.task_type in TASK_TYPES:
            raise ValueError('Unknown task type %s.' % self.task_type)
        if not isinstance(self.metric, _BaseScorer):
            raise ValueError('Metric must be of type scorer')
        if self.mode not in ('fast', 'slow'):
            raise ValueError('Unknown mode %s' % self.mode)

        self._fit(self.predictions, train_labels)
        self._calculate_weights()
        self.identifiers_ = None

        self.base_model_mask = np.full(len(self.predictions), False)
        self.base_model_mask[self.weights_ != 0] = True

        return self

    def _fit(self, predictions, labels):
        if self.mode == 'fast':
            self._fast(predictions, labels)
        else:
            self._slow(predictions, labels)
        return self

    def _fast(self, predictions, labels):
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if self.sorted_initialization:
            n_best = 20 if ensemble_size > 20 else ensemble_size - 1
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = self.calculate_score(pred=ensemble_, y_true=labels)
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                # Memory-efficient averaging!
                ensemble_prediction = np.zeros(ensemble[0].shape)
                for pred in ensemble:
                    ensemble_prediction += pred
                ensemble_prediction /= s

                weighted_ensemble_prediction = (s / float(s + 1)) * ensemble_prediction

            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, pred in enumerate(predictions):
                # TODO: this could potentially be vectorized! - let's profile
                # the script first!
                if self.task_type in CLS_TASKS:
                    fant_ensemble_prediction[:, :] = weighted_ensemble_prediction + (1. / float(s + 1)) * pred
                else:
                    fant_ensemble_prediction[:] = weighted_ensemble_prediction + (1. / float(s + 1)) * pred

                scores[j] = -self.calculate_score(pred=fant_ensemble_prediction, y_true=labels)

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]

    def _slow(self, predictions, labels):
        """Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if self.sorted_initialization:
            n_best = 20 if ensemble_size > 20 else ensemble_size - 1
            indices = self._sorted_initialization(predictions, labels, n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = self.calculate_score(pred=ensemble_, y_true=labels)
                trajectory.append(ensemble_performance)
            ensemble_size -= n_best

        for i in range(ensemble_size):
            scores = np.zeros([predictions.shape[0]])
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                scores[j] = -self.calculate_score(pred=ensemble_prediction, y_true=labels)
                ensemble.pop()
            best = np.nanargmin(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = np.array(order)
        self.trajectory_ = np.array(trajectory)
        self.train_score_ = trajectory[-1]

    def _calculate_weights(self):
        assert len(self.indices_) == self.ensemble_size
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def _sorted_initialization(self, predictions, labels, n_best):
        perf = np.zeros([predictions.shape[0]])

        for idx, prediction in enumerate(predictions):
            perf[idx] = self.calculate_score(pred=predictions, y_true=labels)

        indices = np.argsort(perf)[perf.shape[0] - n_best:]
        return indices

    def predict(self, data, refit='full'):
        predictions = []
        cur_idx = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                if self.base_model_mask[cur_idx]:
                    path = CombinedTopKModelSaver.get_parse_path(path, mode=refit, **self.resampling_params)
                    op_list, estimator, _ = CombinedTopKModelSaver._load(path)
                    predictions.append(fetch_predict_results(self.task_type, op_list, estimator, data))
                cur_idx += 1
        predictions = np.asarray(predictions)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if predictions.shape[0] == len(self.weights_):
            return np.average(predictions, axis=0, weights=self.weights_)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif predictions.shape[0] == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            return np.average(predictions, axis=0, weights=non_null_weights)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")

    def stack_predict(self, features):

        # features shape: num_data, n_base_model*n_dim
        assert features.shape[1] % self.ensemble_size == 0
        assert len(self.weights_) == self.ensemble_size
        n_dim = features.shape[1] // self.ensemble_size

        data_len = features.shape[0]

        features = features.reshape(data_len, -1, n_dim)

        pred = np.average(features, axis=1, weights=self.weights_)

        if self.task_type in CLS_TASKS and n_dim == 1:
            pred = np.hstack([1-pred, pred])

        if pred.shape[1] == 1:
            pred = pred.reshape(-1)

        return pred


    def __str__(self):
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                          for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(self, models):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            model = models[identifier]
            if weight > 0.0:
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self):
        return self.trajectory_[-1]

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, model_path) in enumerate(model_to_eval):
                if self.weights_[model_cnt] != 0:
                    ens_config.append((algo_id, config, model_path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'ensemble_selection'
        ens_info['config'] = ens_config
        ens_info['ensemble_weights'] = [w for w in self.weights_ if w > 0]
        return ens_info
