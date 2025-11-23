import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    对数据集进行预处理，以便用于训练。

    参数:
    - data1: 输入数据集，形状为 (num_examples, num_features)。
    - polynomial_degree: 多项式特征的阶数，默认为 0（不生成多项式特征）。
    - sinusoid_degree: 正弦特征的阶数，默认为 0（不生成正弦特征）。
    - normalize_data: 是否对数据进行标准化，默认为 True。

    返回:
    - data_processed: 预处理后的数据集。
    - features_mean: 数据的均值（如果进行了标准化）。
    - features_deviation: 数据的标准差（如果进行了标准化）。
    """

    # 计算样本总数
    num_examples = data.shape[0]

    # 复制原始数据，避免修改原始数据
    data_processed = np.copy(data)

    # 初始化均值和标准差
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed

    # 数据标准化
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)  # 调用 normalize 函数进行标准化
        data_processed = data_normalized

    # 生成正弦特征
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)  # 调用 generate_sinusoids 函数生成正弦特征
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)  # 将正弦特征拼接到数据中

    # 生成多项式特征
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)  # 调用 generate_polynomials 函数生成多项式特征
        data_processed = np.concatenate((data_processed, polynomials), axis=1)  # 将多项式特征拼接到数据中

    # 在数据前添加一列 1，用于偏置项（bias term）
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    # 返回处理后的数据、均值、标准差
    return data_processed, features_mean, features_deviation