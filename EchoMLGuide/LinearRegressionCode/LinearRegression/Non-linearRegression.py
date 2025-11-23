import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入线性回归模型
from linear_regression import LinearRegression

# 读取数据集
data = pd.read_csv('../data/non-linear-regression-x-y.csv')

# 提取输入特征 x 和输出标签 y，并将其转换为二维数组
x = data['x'].values.reshape((data.shape[0], 1))  # 输入特征 x
y = data['y'].values.reshape((data.shape[0], 1))  # 输出标签 y

# 查看数据集的前 10 行
data.head(10)

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 绘制原始数据的散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Training Data', alpha=0.6)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Original Data Scatter Plot', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 设置训练参数
num_iterations = 500  # 迭代次数
learning_rate = 0.02  # 学习率
polynomial_degree = 15  # 多项式特征的阶数
sinusoid_degree = 15  # 正弦特征的阶数
normalize_data = True  # 是否对数据进行标准化

# 初始化线性回归模型
linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)

# 训练模型
(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

# 输出训练过程中的初始损失值和最终损失值
print('开始损失: {:.2f}'.format(cost_history[0]))  # 初始损失值
print('结束损失: {:.2f}'.format(cost_history[-1]))  # 最终损失值

# 将模型参数 theta 转换为 DataFrame 并打印
theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})
print(theta_table)

# 绘制损失函数随迭代次数的变化图
plt.figure(figsize=(8, 6))
plt.plot(range(num_iterations), cost_history, color='green', linewidth=2)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.title('Gradient Descent Progress', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 生成预测数据
predictions_num = 1000  # 预测点的数量
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1)  # 生成均匀分布的预测点
y_predictions = linear_regression.predict(x_predictions)  # 使用模型进行预测

# 绘制原始数据和预测结果的对比图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Training Data', alpha=0.6)
plt.plot(x_predictions, y_predictions, color='red', linewidth=2, label='Prediction')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Model Prediction vs Training Data', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()