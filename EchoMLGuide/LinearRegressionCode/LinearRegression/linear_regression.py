import numpy as np

from MachineLearning.LinearRegressionCode.utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        初始化线性回归模型。

        参数:
        - data1: 输入特征数据。
        - labels: 目标值（标签）。
        - polynomial_degree: 多项式特征的阶数，默认为0（不使用多项式特征）。
        - sinusoid_degree: 正弦特征的阶数，默认为0（不使用正弦特征）。
        - normalize_data: 是否对数据进行标准化，默认为True。
        """
        # 对数据进行预处理，包括多项式特征生成、正弦特征生成和标准化
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        # 存储处理后的数据、标签、特征的均值和标准差
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 获取特征的数量，并初始化参数矩阵theta为全零矩阵
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        """
        训练模型，执行梯度下降。

        参数:
        - alpha: 学习率。
        - num_iterations: 迭代次数，默认为500。

        返回:
        - theta: 训练得到的参数。
        - cost_history: 每次迭代的损失值记录。
        """
        # 执行梯度下降，并返回最终的参数和损失历史
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际执行梯度下降的迭代过程。

        参数:
        - alpha: 学习率。
        - num_iterations: 迭代次数。

        返回:
        - cost_history: 每次迭代的损失值记录。
        """
        cost_history = []
        # 迭代num_iterations次，每次更新参数并记录损失值
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        执行一次梯度下降的参数更新。

        参数:
        - alpha: 学习率。
        """
        # 获取样本数量
        num_examples = self.data.shape[0]
        # 计算当前参数的预测值
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        # 计算预测值与实际值的误差
        delta = prediction - self.labels
        # 更新参数theta
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        计算损失函数（均方误差）。

        参数:
        - data1: 输入特征数据。
        - labels: 目标值（标签）。

        返回:
        - cost: 损失值。
        """
        # 获取样本数量
        num_examples = data.shape[0]
        # 计算预测值与实际值的误差
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        # 计算均方误差
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        计算假设函数的预测值。

        参数:
        - data1: 输入特征数据。
        - theta: 模型参数。

        返回:
        - predictions: 预测值。
        """
        # 计算预测值：data1 * theta
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        计算给定数据的损失值。

        参数:
        - data1: 输入特征数据。
        - labels: 目标值（标签）。

        返回:
        - cost: 损失值。
        """
        # 对输入数据进行预处理
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        # 计算损失值
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        使用训练好的模型进行预测。

        参数:
        - data1: 输入特征数据。

        返回:
        - predictions: 预测值。
        """
        # 对输入数据进行预处理
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        # 计算预测值
        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions
