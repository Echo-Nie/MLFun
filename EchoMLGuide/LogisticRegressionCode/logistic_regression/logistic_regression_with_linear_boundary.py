import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression


data = pd.read_csv('../data/iris.csv')
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']

x_axis = 'petal_length'
y_axis = 'petal_width'

for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class']==iris_type],
                data[y_axis][data['class']==iris_type],
                label = iris_type
                )
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis,y_axis]].values.reshape((num_examples,2))
y_train = data['class'].values.reshape((num_examples,1))

max_iterations = 1000
polynomial_degree = 0
sinusoid_degree = 0



logistic_regression = LogisticRegression(x_train,y_train,polynomial_degree,sinusoid_degree)
thetas,cost_histories = logistic_regression.train(max_iterations)
labels = logistic_regression.unique_labels

plt.plot(range(len(cost_histories[0])),cost_histories[0],label = labels[0])
plt.plot(range(len(cost_histories[1])),cost_histories[1],label = labels[1])
plt.plot(range(len(cost_histories[2])),cost_histories[2],label = labels[2])
plt.show()

y_train_prections = logistic_regression.predict(x_train)
precision = np.sum(y_train_prections == y_train)/y_train.shape[0] * 100
print ('precision:',precision)

x_min = np.min(x_train[:,0])
x_max = np.max(x_train[:,0])
y_min = np.min(x_train[:,1])
y_max = np.max(x_train[:,1])
samples= 150 
X = np.linspace(x_min,x_max,samples)
Y = np.linspace(y_min,y_max,samples)

Z_SETOSA = np.zeros((samples,samples))
Z_VERSICOLOR = np.zeros((samples,samples))
Z_VIRGINICA = np.zeros((samples,samples))

for x_index,x in enumerate(X):
    for y_index,y in enumerate(Y):
        data = np.array([[x,y]])
        prediction = logistic_regression.predict(data)[0][0]
        if prediction == 'SETOSA':
            Z_SETOSA[x_index][y_index] =1
        elif prediction == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] =1
        elif prediction == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] =1
            
for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(),0],
        x_train[(y_train == iris_type).flatten(),1],
        label = iris_type
                )

plt.contour(X,Y,Z_SETOSA)
plt.contour(X,Y,Z_VERSICOLOR)
plt.contour(X,Y,Z_VIRGINICA)
plt.show()
