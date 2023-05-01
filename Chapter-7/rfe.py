import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

model = LinearRegression()
rfe = RFE(
    estimator=model,
    n_features_to_select=3
)

rfe.fit(X, y)
X_transformed = rfe.transform(X)

"""selected_features = X[:, rfe.support_]
print(selected_features)"""

selected_features_name = [name for name, mask in zip(range(len(X[0])), rfe.support_) if mask]
print(selected_features_name)
