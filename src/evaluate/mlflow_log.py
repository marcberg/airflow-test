import mlflow
from airflow.models import Variable
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, classification_report
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve
import numpy as np
import os

def create_new_experiment(model_type):
    """
    Creates a new MLflow experiment with a unique name based on a base name.
    """
    experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}"

    Variable.set("current_experiment_name", experiment_name)
    mlflow.set_experiment(experiment_name)

def log_model_to_mlflow(pipeline, algorithm_name):
    
    X_train = pd.read_csv('artifacts/X_train.csv')
    y_train = pd.read_csv('artifacts/y_train.csv')
    X_val = pd.read_csv('artifacts/X_val.csv')
    y_val = pd.read_csv('artifacts/y_val.csv')
    
    # Set MLflow experiment
    experiment_name = Variable.get("current_experiment_name")
    mlflow.set_experiment(experiment_name)
    
    # Predict on the training and test sets
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    
    # Calculate metrics for training data
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "train_precision": precision_score(y_train, y_train_pred, average='weighted'),
        "val_precision": precision_score(y_val, y_val_pred, average='weighted'),
        "train_recall": recall_score(y_train, y_train_pred, average='weighted'),
        "val_recall": recall_score(y_val, y_val_pred, average='weighted'),
        "train_f1_score": f1_score(y_train, y_train_pred, average='weighted'),
        "val_f1_score": f1_score(y_val, y_val_pred, average='weighted')
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name=algorithm_name):
        # Log the model
        mlflow.sklearn.log_model(pipeline, "pipeline", registered_model_name=algorithm_name)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model parameters if available
        if hasattr(pipeline, "get_params"):
            params = pipeline.get_params()
            mlflow.log_params(params)


def log_calibration_model_to_mlflow(calib_model, val_score, y_val, test_score, y_test):

    # Set MLflow experiment
    experiment_name = Variable.get("current_experiment_name")
    experiment_name = experiment_name.replace("classification", "calibration")
    mlflow.set_experiment(experiment_name)

    # Apply calibration to val_scores
    val_calibrated_scores = calib_model.predict(val_score.reshape(-1, 1)).flatten()
    test_calibrated_scores = calib_model.predict(test_score.reshape(-1, 1)).flatten()

    # Find the best threshold
    precision, recall, thresholds = precision_recall_curve(y_val, val_calibrated_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # Convert probabilities to class labels using the best threshold
    val_predictions_labels = (val_calibrated_scores > best_threshold).astype(int)
    test_predictions_labels = (test_calibrated_scores > best_threshold).astype(int)

    # Calculate metrics
    metrics = {
        "train_accuracy": accuracy_score(y_val, val_predictions_labels),
        "val_accuracy": accuracy_score(y_test, test_predictions_labels),
        "train_precision": precision_score(y_val, val_predictions_labels, average='weighted'),
        "val_precision": precision_score(y_test, test_predictions_labels, average='weighted'),
        "train_recall": recall_score(y_val, val_predictions_labels, average='weighted'),
        "val_recall": recall_score(y_test, test_predictions_labels, average='weighted'),
        "train_f1_score": f1_score(y_val, val_predictions_labels, average='weighted'),
        "val_f1_score": f1_score(y_test, test_predictions_labels, average='weighted')
    }

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log the calibration model (Keras/TensorFlow model)
        mlflow.tensorflow.log_model(
            model=calib_model,  # Keras/TensorFlow model object
            artifact_path="calibration_model",  # Path to store the model artifact
        )

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Plot and log the artifact
        plt.figure(figsize=(8, 6))
        plt.scatter(val_calibrated_scores, val_score, alpha=0.5, label="Validation")
        plt.scatter(test_calibrated_scores, test_score, alpha=0.5, label="Test")
        plt.xlabel("Calibrated Prediction")
        plt.ylabel("Downsampled Prediction")
        plt.title("Downsampled vs. Calibrated Prediction")
        plt.legend()
        plt.grid(True)

        # Save and log the plot
        plot_path = "artifacts/score_vs_prediction.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)