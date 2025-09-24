# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **Ridge-Regression**
#
# Ridge Regression is a type of linear regression that includes a regularization term to prevent overfitting. It is particularly useful when dealing with multicollinearity in the dataset.
#
# It is based on L2 Regularization, which adds a penalty equal to the square of the magnitude of coefficients to the loss function. 
# This helps to reduce model complexity and prevent overfitting, especially in cases where the number of predictors is large or when predictors are highly correlated.
#
# It is implemented in Python using the `Ridge` class from the `sklearn.linear_model` module.
#
# Rigde regression can be used to predict a continuous target variable based on one or more predictor variables.
# It is commonly used in various fields such as finance, biology, and engineering.

# %%
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# %%
X, y = make_regression(n_samples=10000, n_features=15, noise=0.4, n_informative=6, random_state=42)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)

# %%
y_pred = ridge.predict(X_test_scaled)
display(display(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})))

# %%
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

print("Mean Squared Error 1:", mean_squared_error(y_test, y_pred))
print("R2 Score 1:", r2_score(y_test, y_pred))
print("Mean Absolute Error 1:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error :", root_mean_squared_error(y_test, y_pred))

# %%
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
    'max_iter': [1000, 1500, 2000]
    }

# %%
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=10, n_jobs=2, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# %%
y_pred_2 = grid_search.predict(X_test_scaled)
display(display(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})))
print(grid_search.best_params_)

# %%
print("Mean Squared Error 2:", mean_squared_error(y_test, y_pred_2))
print("R2 Score 2:", r2_score(y_test, y_pred_2))
print("Mean Absolute Error 2:", mean_absolute_error(y_test, y_pred_2))
print("Root Mean Squared Error 2:", root_mean_squared_error(y_test, y_pred_2))

# %%
