import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# 加载数据
data = np.loadtxt("./data2/multiple_variable.txt", delimiter="\t")

# 特征和目标变量
X = data[:, :-1]  # 假设最后一列是目标变量
y = data[:, -1]

# 数据预处理
# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练多变量线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 模型评估
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_r2 = r2_score(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f'Training R² Score: {train_r2:.4f}, Training MSE: {train_mse:.4f}')
print(f'Testing R² Score: {test_r2:.4f}, Testing MSE: {test_mse:.4f}')

# 可视化（假设我们有三个特征，选择前两个特征进行3D展示）
if X.shape[1] >= 2:
    # 创建网格以绘制回归平面
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50))
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    # 如果有更多特征，用均值填充
    if X.shape[1] > 2:
        mean_features = np.mean(X, axis=0)
        X_grid = np.hstack([X_grid, np.tile(mean_features[2:], (X_grid.shape[0], 1))])

    # 预测网格点的值
    y_grid = model.predict(X_grid)
    y_grid = y_grid.reshape(x1_grid.shape)

    # 创建3D散点图和回归平面
    trace1 = go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=y,
        mode='markers',
        marker=dict(
            size=4,
            color=y,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Data Points'
    )

    trace2 = go.Surface(
        x=x1_grid,
        y=x2_grid,
        z=y_grid,
        colorscale='Blues',
        opacity=0.7,
        name='Regression Plane'
    )

    layout = go.Layout(
        title='3D Visualization of Multiple Linear Regression',
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Target Variable'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
        width=1000
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # 保存为HTML文件
    plotly.offline.plot(fig, filename='output/multiple_linear_regression.html')

    # 如果需要显示图表
    fig.show()
else:
    print("需要至少两个特征才能进行3D可视化")