import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 禁用 LaTeX 渲染
plt.rcParams['text.usetex'] = False

# 设置字体样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 导入线性回归模型
from linear_regression import LinearRegression

# 读取数据集
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 将数据集分为训练集和测试集
train_data = data.sample(frac=0.8)  # 80% 的数据作为训练集
test_data = data.drop(train_data.index)  # 剩余 20% 的数据作为测试集

# 定义输入特征和输出标签的列名
input_param_name = 'Economy..GDP.per.Capita.'  # 输入特征：人均 GDP
output_param_name = 'Happiness.Score'  # 输出标签：幸福指数

# 提取训练集和测试集的输入特征和输出标签
x_train = train_data[[input_param_name]].values  # 训练集输入特征
y_train = train_data[[output_param_name]].values  # 训练集输出标签
x_test = test_data[input_param_name].values  # 测试集输入特征
y_test = test_data[output_param_name].values  # 测试集输出标签

# 绘制训练集和测试集的散点图
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label='Train data1', color='blue', alpha=0.6, edgecolor='black')  # 训练集散点图
plt.scatter(x_test, y_test, label='Test data1', color='green', alpha=0.6, edgecolor='black')  # 测试集散点图
plt.xlabel('GDP per Capita', fontsize=14)
plt.ylabel('Happiness Score', fontsize=14)
plt.title('Economy GDP vs Happiness Score', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# 设置训练参数
num_iterations = 500  # 迭代次数
learning_rate = 0.01  # 学习率

# 初始化线性回归模型
linear_regression = LinearRegression(x_train, y_train)

# 训练模型
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

# 输出训练过程中的初始损失值和最终损失值
print('Initial cost:', cost_history[0])  # 初始损失值
print('Final cost:', cost_history[-1])  # 最终损失值

# 绘制损失函数随迭代次数的变化图
plt.figure(figsize=(8, 6))
plt.plot(range(num_iterations), cost_history, color='red', linewidth=2)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.title('Gradient Descent: Cost vs Iterations', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 生成预测数据
predictions_num = 1000  # 预测点的数量
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)  # 生成均匀分布的预测点
y_predictions = linear_regression.predict(x_predictions)  # 使用模型进行预测

# 绘制训练集、测试集和预测结果的对比图
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label='Train data1', color='blue', alpha=0.6, edgecolor='black')  # 训练集散点图
plt.scatter(x_test, y_test, label='Test data1', color='green', alpha=0.6, edgecolor='black')  # 测试集散点图
plt.plot(x_predictions, y_predictions, 'r', label='Prediction', linewidth=2)  # 预测结果曲线
plt.xlabel('GDP per Capita', fontsize=14)
plt.ylabel('Happiness Score', fontsize=14)
plt.title('Economy GDP vs Happiness Score with Prediction', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()