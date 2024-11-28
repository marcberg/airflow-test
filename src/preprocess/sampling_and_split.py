import os 
import pandas as pd 

from sklearn.model_selection import train_test_split

def downsample_and_split(df, target, train_size=0.7, pct_event=0.5):

    X_train, X_tmp, y_train, y_tmp = train_test_split(df.drop([target], axis=1), df[target], train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, train_size=train_size)

    X_train_ones = X_train[y_train == 1]
    y_train_ones = y_train[y_train == 1]

    X_train_zeros = X_train[y_train == 0]
    y_train_zeros = y_train[y_train == 0]

    # Calculate the number of zeros to include for 20% ones in the final dataset
    n_ones = len(y_train_ones)
    n_total = int(n_ones / pct_event) 
    n_zeros = n_total - n_ones

    # Downsample the zeros
    X_train_zeros_downsampled = X_train_zeros.sample(n=n_zeros, random_state=42)
    y_train_zeros_downsampled = y_train_zeros.sample(n=n_zeros, random_state=42)

    # Combine the ones and downsampled zeros
    X_train_downsampled = pd.concat([X_train_ones, X_train_zeros_downsampled])
    y_train_downsampled = pd.concat([y_train_ones, y_train_zeros_downsampled])

    # Save the resulting datasets as CSV files
    output_dir = "artifacts"
    X_train_downsampled.to_csv(f"{output_dir}/X_train.csv", index=False)
    y_train_downsampled.to_csv(f"{output_dir}/y_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)