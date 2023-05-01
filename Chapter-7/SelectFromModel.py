import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

model = RandomForestRegressor()

sfm = SelectFromModel(estimator=model)
X_transformed = sfm.fit_transform(X, y)

support = sfm.get_support()

print([x for x, y in zip(col_names, support) if y == True])