import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import os
# 初始化 Plotly 的笔记本模式
plotly.offline.init_notebook_mode()

# 导入线性回归模型
from linear_regression import LinearRegression

# 读取数据集
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 将数据集分为训练集和测试集
train_data = data.sample(frac=0.8)  # 80% 的数据作为训练集
test_data = data.drop(train_data.index)  # 剩余 20% 的数据作为测试集

# 定义输入特征和输出标签的列名
input_param_name_1 = 'Economy..GDP.per.Capita.'  # 输入特征 1：人均 GDP
input_param_name_2 = 'Freedom'  # 输入特征 2：自由度
output_param_name = 'Happiness.Score'  # 输出标签：幸福指数

# 提取训练集和测试集的输入特征和输出标签
x_train = train_data[[input_param_name_1, input_param_name_2]].values  # 训练集输入特征
y_train = train_data[[output_param_name]].values  # 训练集输出标签
x_test = test_data[[input_param_name_1, input_param_name_2]].values  # 测试集输入特征
y_test = test_data[[output_param_name]].values  # 测试集输出标签

# 配置训练集的 3D 散点图
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),  # 特征 1：人均 GDP
    y=x_train[:, 1].flatten(),  # 特征 2：自由度
    z=y_train.flatten(),  # 输出标签：幸福指数
    name='Training Set',  # 图例名称
    mode='markers',  # 绘制散点图
    marker={
        'size': 10,  # 点的大小
        'opacity': 1,  # 点的透明度
        'line': {
            'color': 'rgb(255, 255, 255)',  # 点的边框颜色
            'width': 1  # 点的边框宽度
        },
    }
)

# 配置测试集的 3D 散点图
plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),  # 特征 1：人均 GDP
    y=x_test[:, 1].flatten(),  # 特征 2：自由度
    z=y_test.flatten(),  # 输出标签：幸福指数
    name='Test Set',  # 图例名称
    mode='markers',  # 绘制散点图
    marker={
        'size': 10,  # 点的大小
        'opacity': 1,  # 点的透明度
        'line': {
            'color': 'rgb(255, 255, 255)',  # 点的边框颜色
            'width': 1  # 点的边框宽度
        },
    }
)

# 配置 3D 图的布局
plot_layout = go.Layout(
    title='Date Sets',  # 图表标题
    scene={
        'xaxis': {'title': input_param_name_1},  # x 轴标题
        'yaxis': {'title': input_param_name_2},  # y 轴标题
        'zaxis': {'title': output_param_name}  # z 轴标题
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}  # 图表边距
)

# 将训练集和测试集的散点图数据合并
plot_data = [plot_training_trace, plot_test_trace]

# 创建图表
plot_figure = go.Figure(data=plot_data, layout=plot_layout)

# 设置训练参数
num_iterations = 500  # 迭代次数
learning_rate = 0.01  # 学习率
polynomial_degree = 0  # 多项式特征的阶数
sinusoid_degree = 0  # 正弦特征的阶数

# 初始化线性回归模型
linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

# 训练模型
(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

# 输出训练过程中的损失值
print('开始损失', cost_history[0])  # 初始损失值
print('结束损失', cost_history[-1])  # 最终损失值

# 绘制损失函数随迭代次数的变化图
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')  # x 轴标签
plt.ylabel('Cost')  # y 轴标签
plt.title('Gradient Descent Progress')  # 图表标题
plt.show()

# 生成预测数据
predictions_num = 10  # 每个特征的预测点数
x_min = x_train[:, 0].min()  # 特征 1 的最小值
x_max = x_train[:, 0].max()  # 特征 1 的最大值
y_min = x_train[:, 1].min()  # 特征 2 的最小值
y_max = x_train[:, 1].max()  # 特征 2 的最大值

# 生成均匀分布的特征值
x_axis = np.linspace(x_min, x_max, predictions_num)  # 特征 1 的预测点
y_axis = np.linspace(y_min, y_max, predictions_num)  # 特征 2 的预测点

# 初始化预测数据的存储数组
x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

# 生成所有可能的特征组合
x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

# 使用模型进行预测
z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

# 配置预测平面的 3D 散点图
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),  # 特征 1 的预测值
    y=y_predictions.flatten(),  # 特征 2 的预测值
    z=z_predictions.flatten(),  # 预测的标签值
    name='Prediction Plane',  # 图例名称
    mode='markers',  # 绘制散点图
    marker={
        'size': 1,  # 点的大小
    },
    opacity=0.8,  # 点的透明度
    surfaceaxis=2,  # 表面轴
)

# 将训练集、测试集和预测平面的数据合并
plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)

os.makedirs('./output', exist_ok=True)
# 指定保存路径和文件名
save_path = './output/temp-LinearRegression-plot.html'
# 在笔记本中显示图表并保存到指定路径
plotly.offline.plot(plot_figure, filename=save_path)