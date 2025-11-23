# MachineLearning

In the process of learning machine learning, individuals summarize the relevant documentation and source code

<br>

## 1 Linear Regression

```bash
LinearRegressionCode/
├── LinearRegression/                  # Directory for linear regression modules
│   ├── MultivariateLinearRegression.py  # Multivariate linear regression
│   ├── Non-linearRegression.py         # Non-linear regression
│   ├── UnivariateLinearRegression.py   # Uni variate linear regression
│   ├── linear_regression.py            # Main script for linear regression
├── LinearRegressionTest/
│   ├── img # Folder containing Jupyter Notebook related files
│   ├── LinearRegressionWithSKLearn.ipynb 
│   # Detailed analysis of each step of linear regression, combined with multiple experiments
├── data1
├── util
```
<br>

## 2 ModelEvaluationMethod
```bash
ModelEvaluationMethod/
├── data1  # Datasets
├── img # Images related to Jupyter Notebooks
├── ModelEvaluationMethod.ipynb 
# Code related to model evaluation methods, learning sklearn
```
<br>

## 3 Logistic
```bash
LogisticRegressionCode/
│
├── data1/
│
├── logistic_regression/
│   ├── logistic_regression.py       # Implementation of the Logistic Regression algorithm
│   ├── logistic_regression_with_linear_boundary.py  # Logistic Regression with linear boundary
│   └── NonLinearBoundary.py         # Handling non-linear boundaries
│
└── util/
    ├── features/              # Utility functions for feature processing
    │   ├── __init__.py         # Initialization
    │   ├── generate_polynomials.py  # Generate polynomial features
    │   ├── generate_sinusoids.py    # Generate sinusoidal features
    │   ├── normalize.py          # Data normalization
    │   └── prepare_for_training.py  # Prepare data for training
    └── hypothesis/             # Utility functions for hypothesis-related calculations
        ├── __init__.py         # Initialization
        ├── sigmoid.py          # Implementation of the sigmoid activation function
        └── sigmoid_gradient.py  # Calculation of the sigmoid gradient
```
