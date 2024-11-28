import joblib
import pandas as pd

from evaluate.mlflow_log import log_model_to_mlflow
from preprocess.feature_engineering import fe_pipeline

def train_final_model(algorithm):

    algorithm_name = type(algorithm).__name__
    best_params = joblib.load(f"artifacts/{algorithm_name}_best_params.pkl")

    pipeline = fe_pipeline(algorithm)
    pipeline.set_params(**best_params)

    X_train = pd.read_csv('artifacts/X_train.csv')
    y_train = pd.read_csv('artifacts/y_train.csv')

    pipeline.fit(X_train, y_train.values.ravel())

    log_model_to_mlflow(pipeline, algorithm_name)

    joblib.dump(pipeline, f"artifacts/{algorithm_name}_final_model.pkl")