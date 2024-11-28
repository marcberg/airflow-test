from skopt import BayesSearchCV
import joblib
import pandas as pd

from preprocess.feature_engineering import fe_pipeline

def bayesian_hyperparameter_tuning(algorithm, param_space, cv=5, n_iter=50, scoring='accuracy', random_state=None):

    # Create a pipeline with standardization and the chosen algorithm
    pipeline = fe_pipeline(algorithm)

    # Prefix the parameter space with 'model__' to target the pipeline
    prefixed_param_space = {f'model__{key}': value for key, value in param_space.items()}
    
    # Set up Bayesian search
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=prefixed_param_space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1
    )
    
    X_train = pd.read_csv('artifacts/X_train.csv')
    y_train = pd.read_csv('artifacts/y_train.csv')
    
    # Fit the search
    bayes_search.fit(X_train, y_train.values.ravel())

    algorithm_name = type(algorithm).__name__
    joblib.dump(bayes_search.best_params_, f"artifacts/{algorithm_name}_best_params.pkl")

