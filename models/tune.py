from hyperopt.pyll.stochastic import quniform
import mlflow
from mlflow.entities.model_registry import registered_model
import mlflow.sklearn
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
import numpy as np

# configure experiment name
mlflow.set_experiment("housing -hpo-experiment")

# prepare data
data = pd.read_csv("data/train.csv")
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.35)

# define search space
search_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
    "max_depth": hp.quniform("max_depth", 5, 20, 1),
    "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
}


# Objective function
def objective(params):
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_samples_split"] = int(params["min_samples_split"])

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        rf = RandomForestRegressor(**params, random_state=42)
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)
        rsme = root_mean_squared_error(y_test, predictions)

        mlflow.log_metric("rsme", rsme)

        # hyperopt objective: minimize loss (RMSE)
        return {"loss": rmse, "status": STATUS_OK}


def run_tuning():
    with mlflow.start_run(run_name="Hyperopt_RF_tuning"):
        trails = Trails()

        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=15,
            trails=trails,
        )

        # log best results
        print(f"Best hyperparameter: {best_params}")
        mlflow.log_params(best_params)

        # final train with best parameters and log as the champion model
        best_rf = RandomForestRegressor(
            n_estimators=int(best_params["n_estimators"]),
            max_depth=int(best_params["max_depth"]),
            min_samples_split=int(best_params["min_samples_split"]),
            random_state=42,
        )

        best_rf.fit(X_train, y_train)

        mlflow.sklearn.load_model(
            sk_model=best_rf,
            artifact_path="best-sklearn-model",
            registered_model_name="housing-rf-champion",
        )


if __name__ == "__main__":
    run_tuning()
    print(f"tuning model is completed")
