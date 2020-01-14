"""
Trains sample SVM on random data. Template for integrating poc library with mlflow.
"""
import os

import click
import mlflow
import mlflow.sklearn
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

load_dotenv()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


def train(train_x, train_y, alpha, l1_ratio):
    # Execute ElasticNet
    elastic_net = ElasticNet(l1_ratio=l1_ratio, alpha=alpha, random_state=42, normalize=True, max_iter=10000)
    elastic_net.fit(train_x, train_y)

    return elastic_net


def validate(elastic_net, x, y):
    predicted_qualities = elastic_net.predict(x)
    rmse, mae = eval_metrics(y, predicted_qualities)
    return rmse, mae


def run_experiment(data, alpha, l1_ratio):
    train_x, train_y, test_x, test_y = data
    with mlflow.start_run():
        elastic_net = train(train_x, train_y, alpha, l1_ratio)
        train_rmse, train_mae = validate(elastic_net, train_x, train_y)
        test_rmse, test_mae = validate(elastic_net, test_x, test_y)
        # Print out metrics
        print("Train  RMSE: %s" % train_rmse)
        print("Test  RMSE: %s" % test_rmse)
        print("Train  MAE: %s" % train_mae)
        print("Test  MAE: %s" % test_mae)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)

        mlflow.sklearn.log_model(elastic_net, "model")


@click.command()
@click.option("--alpha", default=1.0, type=click.FLOAT)
@click.option("--l1_ratio", default=0.5, type=click.FLOAT)
def main(alpha: float, l1_ratio: float):
    csv_url = \
        'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception:
        print("Unable to download training & test CSV, check your internet connection")

    train_set, test_set = train_test_split(data)

    train_x = train_set.drop(["quality"], axis=1)
    test_x = test_set.drop(["quality"], axis=1)
    train_y = train_set[["quality"]]
    test_y = test_set[["quality"]]

    data = train_x, train_y, test_x, test_y

    run_experiment(data, alpha, l1_ratio)


if __name__ == '__main__':
    main()
