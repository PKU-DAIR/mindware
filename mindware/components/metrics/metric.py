from sklearn.metrics._scorer import make_scorer, _BaseScorer
from functools import partial
from sklearn.utils import assert_all_finite
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import type_of_target
import numpy as np


def cls_mse(y_true, y_prob, *, sample_weight=None, pos_label=None):
    """Compute the Brier score loss.

    The smaller the Brier score loss, the better, hence the naming with "loss".
    The Brier score measures the mean squared difference between the predicted
    probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). It can be decomposed as the sum of refinement loss and
    calibration loss.

    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter `pos_label`, which defaults to
    the greater label unless `y_true` is all 0 or all -1, in which case
    `pos_label` defaults to 1.

    Read more in the :ref:`User Guide <brier_score_loss>`.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True targets.

    y_prob : array of shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    pos_label : int or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:

        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.

    Returns
    -------
    score : float
        Brier score loss.

    References
    ----------
    .. [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    >>> brier_score_loss(y_true, y_prob)
    0.037...
    >>> brier_score_loss(y_true, 1-y_prob, pos_label=0)
    0.037...
    >>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
    0.037...
    >>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
    0.0
    """

    assert_all_finite(y_true)
    assert_all_finite(y_prob)
    check_consistent_length(y_true, y_prob, sample_weight)

    y_type = type_of_target(y_true, input_name="y_true")
    if y_type not in ["binary", 'multiclass']:
        raise ValueError(
            "Only binary classification is supported. The type of the target "
            f"is {y_type}."
        )

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")
    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")

    if len(y_prob.shape) == 1:
        y_prob = y_prob.reshape(-1, 1)
        y_prob = np.hstack((1-y_prob, y_prob))

    true_label = np.zeros(y_prob.shape)
    true_label[range(true_label.shape[0]), y_true] = 1

    diff = np.sum((true_label - y_prob)**2, axis=1)

    return np.average(diff, weights=sample_weight)


def get_metric(metric):
    # Metrics for classification
    if metric in ["accuracy", "acc"]:
        from sklearn.metrics import accuracy_score
        return make_scorer(accuracy_score)
    elif metric in ["balanced_accuracy", "bal_acc"]:
        from sklearn.metrics._scorer import balanced_accuracy_scorer
        return balanced_accuracy_scorer
    elif metric == 'f1':
        from sklearn.metrics import f1_score
        return make_scorer(partial(f1_score, average='macro'))
    elif metric == 'precision':
        from sklearn.metrics import precision_score
        return make_scorer(precision_score)
    elif metric == 'recall':
        from sklearn.metrics import recall_score
        return make_scorer(recall_score)
    elif metric == "auc":
        from sklearn.metrics import roc_auc_score
        return make_scorer(roc_auc_score, needs_threshold=True)
    elif metric in ['log_loss', 'cross_entropy']:
        from sklearn.metrics import log_loss
        return make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    elif metric in ['brier_score_loss']:
        from sklearn.metrics import brier_score_loss
        return make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
    elif metric == 'cls_mse':
        return make_scorer(cls_mse, greater_is_better=False, needs_proba=True)

    # Metrics for regression
    elif metric in ["mean_squared_error", "mse"]:
        from sklearn.metrics import mean_squared_error
        return make_scorer(mean_squared_error, greater_is_better=False)
    elif metric == "rmse":
        from .rgs_metrics import rmse
        return make_scorer(rmse, greater_is_better=False)
    elif metric in ['mean_squared_log_error', "msle"]:
        from sklearn.metrics import mean_squared_log_error
        return make_scorer(mean_squared_log_error, greater_is_better=False)
    elif metric == "evs":
        from sklearn.metrics import explained_variance_score
        return make_scorer(explained_variance_score)
    elif metric == "r2":
        from sklearn.metrics import r2_score
        return make_scorer(r2_score)
    elif metric == "max_error":
        from sklearn.metrics import max_error
        return make_scorer(max_error, greater_is_better=False)
    elif metric in ["mean_absolute_error", "mae"]:
        from sklearn.metrics import mean_absolute_error
        return make_scorer(mean_absolute_error, greater_is_better=False)
    elif metric == "median_absolute_error":
        from sklearn.metrics import median_absolute_error
        return make_scorer(median_absolute_error, greater_is_better=False)
    elif isinstance(metric, _BaseScorer):
        return metric
    elif callable(metric):
        import warnings
        warnings.warn("metric receives a callable and we consider to maximize it!")
        return make_scorer(metric)
    else:
        raise ValueError("Given", str(metric), ". Expect a str or a sklearn.Scorer or a callable")
