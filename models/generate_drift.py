import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import os

data = fetch_california_housing()
X, y = data.data, data.target

# add noise → simulate drift
X_drifted = X + np.random.normal(0, 5, X.shape)

model = LinearRegression()
model.fit(X_drifted, y)

os.makedirs("models/saved", exist_ok=True)
joblib.dump(model, "models/saved/model_v2.pkl")

print("Drifted model saved!")
