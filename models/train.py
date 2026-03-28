import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# cofigure MlFlow Experiment
mlflow.set_experiment("housing-price-prediction")

# prepare data
data = pd.read_csv("data/train.csv")
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


def train_model():
    n_estimators = 100
    max_depth = 10

    with mlflow.start_run(run_name="RandomForest_Basile"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train Model
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        # Evaluate
        predictions = rf.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        # log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        print(f"Model trained. RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # log model & Versioning
        # this saves model and creates a version in the MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=rf, artifact_path="sklean-model", registered_model_name="housing-rf-model"
        )


if __name__ == "__main__":
    train_model()
    print(f"Model trained sucessfully")
# print(f"X \n {X.shape}")
# print(f"y \n {y.shape}")
#   print(f"shape of data is {data.shape}")
#   print(data.head(5))
#   print(f"X_train: {X_train.shape}")
#   print(f"X_test: {X_test.shape}")
#   print(f"y_train: {y_train.shape}")
#   print(f"y_test: {y_test.shape}")
