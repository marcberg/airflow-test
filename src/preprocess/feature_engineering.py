from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Example pipeline
def fe_pipeline(algorithm):
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', algorithm)  
    ])

    return pipeline