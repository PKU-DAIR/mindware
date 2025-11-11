import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.spatial
from sklearn.metrics._scorer import _BaseScorer
from mindware.components.utils.constants import CLS_TASKS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def outliers_mask_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_bound = q3 + 2 * iqr
    mask = data < upper_bound
    return mask

def choose_base_models_regression(predictions, labels, num_model, ratio = 0.49):
    n, s = predictions.shape  # 矩阵维度

    if ratio == -1: # 随机
        z = np.zeros(n, dtype=int)
        top_k_indices = np.random.choice(n, num_model, replace=False)
        z[top_k_indices] = 1
        return z
    
    # y - f_h(x)
    dif = labels - predictions
    _G = dif @ dif.T 
    _G = _G / s
    mask = outliers_mask_iqr(np.diag(_G))
    _G = _G[mask][:, mask]
    G = _G
    # G_min = 0
    # G_max = np.max(G)
    # G = (G - G_min) / (G_max - G_min)
    # G = (G - np.mean(G)) / np.std(G)
    
    # G_max = np.max(np.diag(_G))
    # G = np.zeros_like(_G)
    # for i in range(G.shape[0]):
    #     for j in range(G.shape[1]):
    #         if i == j:
    #             G[i, j] = _G[i, j] / G_max
    #         else:
    #             G[i, j] = 0.5 * (_G[i, j] / _G[i, i] + _G[i, j] / _G[j, j])
    
    I = np.eye(G.shape[0])
    G = ratio * (G * (1 - I)) + (1 - ratio) * (G * I) + 1e-5 * I  # / (2*(num_model-1)) 

    if not np.allclose(G, G.T, atol=1e-8):  # 设置容差
        print("G 不是对称的!")

    # G_ij 为div i,j G_ii 为mse，再调用SDP求解器
    z = cp.Variable(len(G))    # 向量变量
    objective = cp.Minimize(cp.quad_form(z, G))
    constraints = [
        cp.sum(z) == num_model,  # 线性约束 1
        z >= 0,                 # 半正定性约束
        z <= 1
    ]
    problem = cp.Problem(objective, constraints)
    # 求解问题
    problem.solve(solver=cp.CLARABEL)  # 使用 SCS 求解器（也可以选择 MOSEK 等）
    z_value = np.array(z.value)
    top_k_indices = np.argsort(z_value)[-num_model:]

    # sel_G = _G[top_k_indices][:, top_k_indices]

    mask_idx = np.where(mask)[0]
    for i in top_k_indices:
        print((mask_idx[i],G[i][i] / (1 - ratio)))

    top_k_indices = mask_idx[top_k_indices]
    z = np.zeros(n, dtype=int)
    z[top_k_indices] = 1

    return z # , sel_G


def choose_base_models_classification(predictions, labels, num_model, ratio = 0.49):
    if len(labels.shape) == 1:
        labels = OneHotEncoder().fit_transform(np.reshape(labels, (len(labels), 1))).toarray()
    n, s, c = predictions.shape  # 矩阵维度 n:模型数量, c:类别

    if ratio == -1: # 随机
        z = np.zeros(n, dtype=int)
        top_k_indices = np.random.choice(n, num_model, replace=False)
        z[top_k_indices] = 1
        return z
    
    # y - f_h(x)
    dif = labels - predictions
    _G = np.zeros((c, n, n))
    for i in range(c):
        _dif = dif[:, :, i]
        _G[i] = _dif @ _dif.T  # 计算差异值矩阵
    _G = np.sum(_G, axis=0) / s
    diag = np.diag(_G)
    mask = outliers_mask_iqr(diag)
    # 筛掉还不如纯随机预测的模型
    rand_score = (1/c)**2 * (c-1) + (1-1/c)**2
    mask[diag > rand_score] = False
    _G = _G[mask][:, mask]
    G = _G

    I = np.eye(G.shape[0])
    G = ratio * (G * (1 - I)) + (1 - ratio) * (G * I) + 1e-5 * I  # / (2*(num_model-1)) 

    if not np.allclose(G, G.T, atol=1e-8):  # 设置容差
        print("G 不是对称的!")

    # G_ij 为div i,j G_ii 为mse，再调用SDP求解器
    z = cp.Variable(len(G))    # 向量变量
    objective = cp.Minimize(cp.quad_form(z, G))
    constraints = [
        cp.sum(z) == num_model,  # 线性约束 1
        z >= 0,                 # 半正定性约束
        z <= 1
    ]
    problem = cp.Problem(objective, constraints)
    # 求解问题
    problem.solve(solver=cp.CLARABEL)  # 使用 SCS 求解器（也可以选择 MOSEK 等）
    z_value = np.array(z.value)
    top_k_indices = np.argsort(z_value)[-num_model:]

    # sel_G = _G[top_k_indices][:, top_k_indices]

    mask_idx = np.where(mask)[0]
    for i in top_k_indices:
        print((mask_idx[i], G[i][i]/ (1 - ratio)))

    top_k_indices = mask_idx[top_k_indices]
    z = np.zeros(n, dtype=int)
    z[top_k_indices] = 1

    return z # , sel_G

# def choose_base_models_regression(predictions, labels, num_model):
#     base_mask = [0] * len(predictions)
#     dif = predictions - labels
#     dif[dif > 0] = 1
#     dif[dif < 0] = -1
#     '''Calculate the distance between each model'''
#     dist = scipy.spatial.distance.cdist(dif, dif)
#     total_dist = np.sum(dist, 1)
#     '''Select the model which has large distance to other models'''
#     selected_models = total_dist.argsort()[-num_model:]
#     for model in selected_models:
#         base_mask[model] = 1
#     return base_mask

# def choose_base_models_classification(predictions, num_model, interval=20):
#     num_class = predictions.shape[2]
#     num_total_models = predictions.shape[0]
#     base_mask = np.full(len(predictions), False)
#     bucket = np.arange(interval + 1) / interval
#     bucket[0] -= 1e-8
#     bucket[-1] += 1e-8
#     distribution = []
#     for prediction in predictions:
#         freq_array = []
#         for i in range(num_class):
#             class_i = prediction[:, i]
#             group = pd.cut(class_i, bucket, right=False)
#             counts = group.value_counts()
#             freq = list(counts / counts.sum())
#             freq_array += freq

#         # TODO: Debug inf output
#         # print(prediction)
#         # print(freq_array)
#         distribution.append(freq_array)  # Shape: (num_total_models,20*num_class)

#     distribution = np.array(distribution)

#     # Apply the clustering algorithm
#     model = AgglomerativeClustering(n_clusters=num_model, linkage="complete")
#     cluster = model.fit(distribution)
#     """
#     Select models which are the most nearest to the clustering center
#     selected_models = []
#     """
#     for cluster_label in range(num_model):
#         cluster_center = np.zeros(distribution.shape[1])
#         count = 0
#         """
#          Averaging the distribution which belong the same clustering class
#           and then get the corresponding distribution center
#         """
#         for i in range(num_total_models):
#             if cluster.labels_[i] == cluster_label:
#                 count += 1
#                 cluster_center += distribution[i]
#         cluster_center = cluster_center / count
#         distances = np.sqrt(np.sum(np.asarray(cluster_center - distribution) ** 2, axis=1))
#         selected_model = distances.argmin()
#         base_mask[selected_model] = 1

#     return base_mask


def calculate_weights(predictions, labels, base_mask):
    num_total_models = predictions.shape[0]
    num_samples = predictions.shape[1]
    weights = np.zeros((num_samples, num_total_models))
    for i in range(num_total_models):
        if base_mask[i] != 0:
            predicted_labels = np.argmax(predictions[i], 1)
            acc = accuracy_score(predicted_labels, labels)
            model_weight = 0.5 * np.log(acc / (1 - acc))  # a concrete value
            shannon_ent = -1.0 * np.sum(predictions[i] * np.log2(predictions[i]), 1)  # shape: (1, num_samples)
            confidence = 1 / np.exp(shannon_ent)
            model_weight = model_weight * confidence  # The weight of current model to all samples
            model_weight = model_weight.reshape(num_samples, 1)
            weights[:, i] = model_weight
    return weights


def calculate_weights_simple(predictions, labels, base_mask):
    num_total_models = predictions.shape[0]
    weights = [0] * num_total_models
    for i in range(num_total_models):
        if base_mask[i] != 0:
            predicted_labels = np.argmax(predictions[i], 1)
            acc = accuracy_score(predicted_labels, labels)
            model_weight = 0.5 * np.log(acc / (1 - acc))  # a concrete value
            weights[i] = model_weight
    return weights


class UnnamedEnsemble:
    def __init__(
            self,
            ensemble_size: int,
            task_type: int,
            metric: _BaseScorer,
            random_state: np.random.RandomState = None,
    ):
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state
        self.base_model_mask = None
        self.weights_ = None

    def fit(self, predictions, labels):
        """

        :param predictions: proba_predictions for cls. Shape: (num_models,num_samples,num_class) for cls
        :param labels: Shape: (num_samples,)
        :return: self
        """
        if self.task_type in CLS_TASKS:  # If classification
            self.base_model_mask = choose_base_models(predictions, labels, self.ensemble_size)
            self.weights_ = calculate_weights(predictions, labels, self.base_model_mask)
        else:
            pass
        return self

    def predict(self, predictions):
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
