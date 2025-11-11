import numpy as np

def generate_pairwise_data(A, B, S):
    X = []
    Y = []
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            X.append(np.concatenate([A[i], B[j]]))
            Y.append(S[i, j])
    return np.array(X), np.array(Y)

# 该函数用于计算A，B两个序列中排序关系正确的偏序对比例


def calculate_relative(A, B):
    P = 0
    l = len(A)
    for i in range(l):
        for j in range(i):
            if (A[i] <= A[j] and B[i] <= B[j]) or (A[i] > A[j] and B[i] > B[j]):
                P += 1
    return 2 * P / (l * (l - 1))

def train_model(src_meta_feature, sim):
    """
    该函数根据输入的任务meta_feature和相似度ground_truth信息训练预测相似度的CatBoost模型

    src_meta_feature是一个n * l的numpy array, 
        每行表示一个任务的meta feature, l是meta feature维数
    sim是一个n * n的numpy array, 
        表示训练使用的任务两两间相似度的ground truth信息

    注意src_meta_feature和sim中的任务顺序需要对齐
    """
    from catboost import CatBoostRegressor
    train_X, train_Y = generate_pairwise_data(
        src_meta_feature, src_meta_feature, sim)
    surrogate_cat = CatBoostRegressor()
    surrogate_cat.fit(train_X, train_Y, silent=True)

    surrogate_cat.save_model('model.cbm')  # 保存训练得到的模型

    return surrogate_cat
