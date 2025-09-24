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
# # **ML-Pipeline**
#
# ML Pipeline is a way to streamline the process of building and deploying machine learning models. 
#
# It involves a series of steps that include data preprocessing, feature engineering, model training, and evaluation. 
#
# The goal is to create a repeatable and efficient workflow that can be easily managed and scaled.
#
# Pipelines help in automating the workflow of machine learning tasks, making it easier to manage and scale the process.
#
# They allow for the encapsulation of multiple steps in a single object, which can be easily reused and modified.
#
# Pipelines can also help in avoiding data leakage by ensuring that the same transformations are applied to both training and test data.
#
# Pipelines also help in organizing the code and making it more readable.
# They allow you to define a sequence of steps that can be executed in a single command.
# This is particularly useful when you have multiple preprocessing steps or when you want to apply the same preprocessing steps to both training and testing data.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# %%
import pandas as pd
import numpy as np

num_items = 50000
genres = ['Rock', 'Metal', 'Bluegrass']

# 1. Generate clean features and target
followers = np.random.randint(50000, 5000000, num_items)
genre_col = np.random.choice(genres, num_items)

genre_score = pd.Series(genre_col).map({'Rock': 5, 'Metal': 7, 'Bluegrass': 2})
follower_score = (followers / 5000000) * 10
sold_out = (genre_score + follower_score > 10).astype(int)

# 2. Create the clean DataFrame
df = pd.DataFrame({
    'Genre': genre_col,
    'Social_media_followers': followers.astype(float), # Set as float for NaNs
    'Sold_out': sold_out
})

# 3. Inject NaNs directly into the DataFrame
# Randomly select a fraction of rows to set to NaN for each column
df.loc[df.sample(frac=0.11).index, 'Social_media_followers'] = np.nan
df.loc[df.sample(frac=0.22).index, 'Genre'] = np.nan

# 4. Show the final result
print("Dataset 1 Head:")
print(df.head())

print("\nTotal Missing Values:")
print(df.isnull().sum())

# %%
df = df.drop(columns=['Genre'])
df.head()

# %%
df = df.dropna(subset=['Social_media_followers', 'Sold_out'])

# %%
X = df.drop(['Sold_out'], axis=1)
y = df['Sold_out']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)   

# %%
display(X_train.shape, y_train.shape)
display(X_test.shape, y_test.shape)

# %%
from sklearn.impute import SimpleImputer

# Create an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')

# %%
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()

# %%
pipe1 = make_pipeline(imputer, logr)
pipe1.fit(X_train, y_train)

# %%
print("Pipe Final Test 1: ", pipe1.score(X_test, y_test))

# %%
print("Pipe Final Train 1: ", pipe1.score(X_train, y_train))

# %%
# Display the statistics of the imputer
# This will show the mean values used to fill missing data
# This is useful to understand how the imputer has transformed the data
pipe1.named_steps.simpleimputer.statistics_

# %%
# Display the coefficients of the logistic regression model
# This will show the importance of each feature in the model
# This is useful to understand how the model is making predictions
pipe1.named_steps.logisticregression.coef_

# %% [markdown]
# **Advanced Pipeline**
#
# Implementing advanced pipeline with both numerical and categorical data.

# %%
import pandas as pd
import numpy as np

num_items = 50000
genres = ['Rock', 'Metal', 'Bluegrass']

# 1. Generate clean features and target
followers = np.random.randint(50000, 5000000, num_items)
genre_col = np.random.choice(genres, num_items)

genre_score = pd.Series(genre_col).map({'Rock': 5, 'Metal': 7, 'Bluegrass': 2})
follower_score = (followers / 5000000) * 10
sold_out = (genre_score + follower_score > 10).astype(int)

# 2. Create the clean DataFrame
df2 = pd.DataFrame({
    'Genre': genre_col,
    'Social_media_followers': followers.astype(float), # Set as float for NaNs
    'Sold_out': sold_out
})

# 3. Inject NaNs directly into the DataFrame
# Randomly select a fraction of rows to set to NaN for each column
df2.loc[df.sample(frac=0.11).index, 'Social_media_followers'] = np.nan
df2.loc[df.sample(frac=0.22).index, 'Genre'] = np.nan

# 4. Show the final result
print("Dataset 2 Head:")
print(df2.head())

print("\nTotal Missing Values:")
print(df2.isnull().sum())

# %%
df2 = df2.dropna(subset=['Social_media_followers', 'Sold_out'])

# %%
X2 = df2.iloc[:, 0:2]
y2 = df2.iloc[:, 2]

# %%
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=50)

# %%
num_col = ['Social_media_followers']
cat_col = ['Genre']

# %%
# Create pipelines for numerical and categorical data preprocessing
# The numerical pipeline will handle missing values and scale the data
# The categorical pipeline will handle missing values and apply one-hot encoding
# This allows us to preprocess different types of data in a consistent manner
# This is useful for building robust machine learning models that can handle various data types
num_pipeline = Pipeline(steps = [
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline(steps = [
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# %%
from sklearn.compose import ColumnTransformer, make_column_transformer

# Create a column transformer to apply the numerical and categorical pipelines
# This allows us to apply different preprocessing steps to different columns in the dataset
# The `remainder='drop'` argument ensures that any columns not specified in the transformers are dropped from the final output
# This is useful for creating a clean and consistent dataset for training machine learning models
# It is also useful for building robust machine learning models that can handle various data types
# The `n_jobs=-1` argument allows the transformer to use all available CPU cores for parallel processing, speeding up the transformation process
col_transformer = ColumnTransformer(
    transformers = [
    ('num_pipeline', num_pipeline, num_col),
    ('cat_pipeline', cat_pipeline, cat_col)
    ],
    remainder='passthrough',  # Drop any columns not specified in the transformers
    n_jobs = -1  # Use all available CPU cores for parallel processing
)

# %%
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=8, random_state=50)
pipe_final = make_pipeline(col_transformer, dtc)
pipe_final.fit(X2_train, y2_train)

# %%
print("Pipe Final Test 2: ", pipe_final.score(X2_test, y2_test))

# %%
print("Pipe Final Train 2: ", pipe_final.score(X2_train, y2_train))

# %% [markdown]
# **Saving the Pipeline**
#
# Saving the Pipeline is useful for later use, allowing us to reuse the trained model without retraining

# %%
import joblib

# Save the trained pipeline to a file
joblib.dump(pipe_final, 'music_genre_pipeline.joblib')

# %%
pipe_final_2 = joblib.load('music_genre_pipeline.joblib')

# %%
