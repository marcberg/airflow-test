import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from score.score_classification import score_classification
from evaluate.mlflow_log import log_calibration_model_to_mlflow

def train_calibration_model():
    
    X_train, y_train, X_test, y_test = score_classification()

    # Build the neural network model
    model = Sequential([
        Dense(16, activation='relu', input_shape=(1,)),  # Single feature input
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for probability output
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=20, 
                        batch_size=32, 
                        verbose=1)

    log_calibration_model_to_mlflow(calib_model=model,
                                    val_score=X_train, 
                                    y_val=y_train,
                                    test_score=X_test, 
                                    y_test=y_test
                                    )
