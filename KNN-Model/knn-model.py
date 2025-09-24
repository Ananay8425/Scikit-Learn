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
# # **KNN Model**
#
# KNN (K-Nearest Neighbors) is a supervised learning algorithm that can be used for classification or regression tasks. 
#
# It works by finding the K most similar instances in the training data to a new instance and using their labels to make a prediction.
#
# KNN is a simple and intuitive algorithm, but it can be computationally expensive for large datasets. 
# It is also sensitive to the choice of K, which can affect the accuracy of the model. 
#
# Use cases for KNN include:
# - Image classification
# - Recommendation systems
# - Time series forecasting
#

# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

# %%
df = pd.read_csv("https://github.com/RyanNolanData/YouTubeData/blob/main/500hits.csv?raw=true", encoding="latin-1")

# %%
df.head()

# %%
df = df.drop(columns= ['PLAYER', 'CS'])
df.head()

# %%
X = df.iloc[:, 0:13]
y = df.iloc[:, 13]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=45)

# %%
# Initialize the MinMaxScaler
# This will scale the features to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# %%
X_train = scaler.fit_transform(X_train)

# %%
X_test = scaler.fit_transform(X_test)

# %%
# Initialize the KNN classifier
# Using 8 neighbors 
# This is a common choice for KNN
# Using Euclidean distance as the metric, which is standard for KNN
# Euclidean distance is the most common distance metric used in KNN that measures the straight-line distance between two points in Euclidean space.
knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean')

# %%
knn.fit(X_train, y_train)

# %%
Y_pred = knn.predict(X_test)
print(Y_pred)

# %%
print("KNN Score:", knn.score(X_test, y_test))
# The accuracy of the KNN model on the test set
# Accuracy is the ratio of correctly predicted instances to the total instances in the test set.
print("Accuracy:", accuracy_score(y_test, Y_pred))
# Confusion matrix shows the number of correct and incorrect predictions for each class
print("Confusion Matrix:\n", confusion_matrix(y_test, Y_pred))
# Classification report provides precision, recall, f1-score for each class
# It gives a detailed performance evaluation of the model.
print("Classification Report:\n", classification_report(y_test, Y_pred))

# %%
print(knn.n_samples_fit_)

# %%
