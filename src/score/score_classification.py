import pandas as pd
from evaluate.select_best_algorithm import best_classification_model

def score_classification():

    X_val = pd.read_csv('artifacts/X_val.csv')
    y_val = pd.read_csv('artifacts/y_val.csv')
    
    X_test = pd.read_csv('artifacts/X_val.csv')
    y_test = pd.read_csv('artifacts/y_val.csv')

    pipeline = best_classification_model()

    val_scores = pipeline.predict_proba(X_val)
    test_scores = pipeline.predict_proba(X_test)

    return val_scores[:,1], y_val, test_scores[:,1], y_test