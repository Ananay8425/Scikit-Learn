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

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# %%
df = pd.read_csv('https://github.com/codebasics/py/blob/master/ML/1_linear_reg/homeprices.csv?raw=true')
df.head()
df.tail()

# %%
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.area, df.price)

# %%
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# %%
reg.predict([[3300]])

# %%
# In the formula y = mx + b, m is the slope and b is the intercept
# Slope (m) = reg.coef_ and Intercept (b) = reg.intercept_
# y = price, x = area

display(reg.coef_) 
display(reg.intercept_)

sample_price = 135.78767123 * 3300 + 180616.43835616432
print("Price(y) : ", sample_price)

# %%
# Generating a new column for the predicted price with provided area in new csv file
df2 = pd.read_csv('https://github.com/codebasics/py/blob/master/ML/1_linear_reg/areas.csv?raw=true')
df2

# %%
pr2 = reg.predict(df2)
display(pr2)

# %%
df2.to_csv('area_predicted_price.csv')
display(df2)

# %%
plt.xlabel('Area')
plt.ylabel('Predicted Price')
plt.scatter(df2.area, pr2, marker='*', color="red")

# %%
