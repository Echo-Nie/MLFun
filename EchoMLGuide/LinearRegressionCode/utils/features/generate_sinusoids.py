import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    生成正弦特征。

    参数:
    - uav: 输入数据集，形状为 (num_examples, num_features)。
    - sinusoid_degree: 正弦特征的阶数，决定生成多少个不同频率的正弦特征。

    返回:
    - sinusoids: 生成的正弦特征矩阵，形状为 (num_examples, sinusoid_degree * num_features)。
    """

    # 获取数据集的样本数量
    num_examples = dataset.shape[0]

    # 初始化一个空的正弦特征矩阵，列数为 0
    sinusoids = np.empty((num_examples, 0))

    # 遍历每个阶数，生成对应频率的正弦特征
    for degree in range(1, sinusoid_degree + 1):
        # 计算当前阶数的正弦特征：sin(degree * uav)
        sinusoid_features = np.sin(degree * dataset)

        # 将当前阶数的正弦特征拼接到总的正弦特征矩阵中
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    # 返回生成的正弦特征矩阵
    return sinusoids