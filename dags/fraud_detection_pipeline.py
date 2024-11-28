from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess.sampling_and_split import downsample_and_split
from train.hyperparametertuning import bayesian_hyperparameter_tuning
from evaluate.mlflow_log import create_new_experiment
from train.train_with_best_params import train_final_model
from train.calibration_model import train_calibration_model

# Algorithm definitions
algorithms = {
    "RandomForest": {
        "algorithm": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": (10, 200),
            "max_depth": (3, 20),
            "min_samples_split": (2, 10)
        }
    },
    "LogisticRegression": {
        "algorithm": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "C": (0.1, 10.0, 'log-uniform'),
            "penalty": ['l2']
        }
    },
    "LightGBM": {
        "algorithm": LGBMClassifier(random_state=42),
        "params": {
            "num_leaves": (20, 150),
            "learning_rate": (0.01, 0.2, 'log-uniform'),
            "n_estimators": (50, 200),
            "max_depth": (3, 15)
        }
    },
    "XGBoost": {
        "algorithm": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "params": {
            "n_estimators": (50, 200),
            "learning_rate": (0.01, 0.2, 'log-uniform'),
            "max_depth": (3, 15),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0)
        }
    },
    "DecisionTree": {
        "algorithm": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": (3, 20),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 5)
        }
    }
}

# DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}


with DAG(
    dag_id="ml_project_pipeline",
    default_args=default_args,
    description="An Airflow DAG for training ML models",
    schedule_interval="12 */3 * * *",  # Every 3 hours at minute 10
    start_date=datetime(2024, 11, 28, 8, 12),  # Align the start date correctly with schedule
    catchup=False,  # Disable backfilling
    tags=["ml", "classification"],
) as dag:

    # Task 1: Preprocess Data
    def preprocess_data():
        df = pd.read_csv("data/creditcard.csv")
        downsample_and_split(df=df, target="Class")

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    # Task 2: Hyperparameter Tuning
    def tune_model(algorithm_name, algorithm, params):
        bayesian_hyperparameter_tuning(algorithm, params)

    tuning_tasks = []
    for algorithm_name, config in algorithms.items():
        tuning_task = PythonOperator(
            task_id=f"tune_{algorithm_name}",
            python_callable=tune_model,
            op_kwargs={
                "algorithm_name": algorithm_name,
                "algorithm": config["algorithm"],
                "params": config["params"]
            }
        )
        tuning_tasks.append(tuning_task)

    # Task 3: Create Experiment
    create_experiment_task = PythonOperator(
        task_id="create_experiment",
        python_callable=create_new_experiment,
        op_kwargs={"model_type": "classification"}
    )

    # Task 4: Train Final Models
    def train_model(algorithm):
        train_final_model(algorithm)

    training_tasks = []
    for algorithm_name, config in algorithms.items():
        training_task = PythonOperator(
            task_id=f"train_{algorithm_name}",
            python_callable=train_model,
            op_kwargs={"algorithm": config["algorithm"]}
        )
        training_tasks.append(training_task)

    # Task 5: Train Calibration Model
    train_calibration_task = PythonOperator(
        task_id="train_calibration_model",
        python_callable=train_calibration_model
    )

    preprocess_task >> tuning_tasks  # Preprocessing is followed by tuning tasks
    for tuning_task in tuning_tasks:
        tuning_task >> create_experiment_task  # All tuning tasks must complete before creating the experiment
    create_experiment_task >> training_tasks  # Creating experiment is followed by training tasks
    for training_task in training_tasks:
        training_task >> train_calibration_task  # Training tasks are followed by calibration

    ## Task dependencies
    #preprocess_task >> create_experiment_task
    #create_experiment_task >> tuning_tasks
    #for tuning_task in tuning_tasks:
    #    for training_task in training_tasks:
    #        tuning_task >> training_task
    #for training_task in training_tasks:
    #    training_task >> train_calibration_task

