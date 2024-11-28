from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model
import mlflow
import pandas as pd
from airflow.models import Variable

from preprocess.feature_engineering import fe_pipeline

def best_classification_model():
    
    # Get experiment details
    experiment_name = Variable.get("current_experiment_name")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Fetch and sort runs by your key metric
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_run = runs.sort_values("metrics.val_f1_score", ascending=False).iloc[0]

    # Load the best model
    best_run_id = best_run["run_id"]
    model_uri = f"runs:/{best_run_id}/pipeline"
    best_pipeline = load_model(model_uri)

    return best_pipeline