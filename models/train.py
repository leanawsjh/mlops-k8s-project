from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse

data = fetch_california_housing()
X, y = data.dat a, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model v1 R2: {score}")

os.makedirs("models/saved", exist_ok=True)
joblib.dump(model, "models/saved/model_v1.pkl")
