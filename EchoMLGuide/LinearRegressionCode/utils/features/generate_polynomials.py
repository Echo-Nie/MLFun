"""Add polynomial features to the features set"""

import numpy as np
from .normalize import normalize  # 导入归一化函数


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """生成多项式特征
    变换方法：
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    """

    # 将数据集按列分成两部分
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]  # 第一部分特征
    dataset_2 = features_split[1]  # 第二部分特征

    # 获取两部分的形状（样本数，特征数）
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # 检查两部分样本数是否一致
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    # 检查两部分是否有特征
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    # 如果某一部分没有特征，则用另一部分代替
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # 取两部分特征数的最小值
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]  # 截取前 num_features 列
    dataset_2 = dataset_2[:, :num_features]  # 截取前 num_features 列

    # 初始化一个空的多项式特征矩阵
    polynomials = np.empty((num_examples_1, 0))

    # 生成多项式特征
    for i in range(1, polynomial_degree + 1):  # 遍历多项式阶数
        for j in range(i + 1):  # 遍历当前阶数的组合
            # 计算多项式特征
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            # 将生成的特征拼接到多项式特征矩阵中
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # 如果需要归一化，则对多项式特征进行归一化
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    # 返回生成的多项式特征
    return polynomials