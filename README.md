# Description

This project is to test setting up airflow on my windows computer using Docker. 

What I will be running in airflow is a simple ML project where I do:
- preprocessing (standardize and down sampling).
- hyperparameter tuning with different algorithms.
- train models and log to mlflow.
- calibrate score (because of the down sampling) and log to mlflow.

So I also want to access 

# Setup

For developing

```
conda create --name venv_airflow_test python=3.9 -y

conda activate venv_airflow_test

conda install ipykernel -y

pip install -r requirements.txt
```

Build and start the Airflow environment:

```
docker-compose up --build
```

If making changes to the python code (not requirments, folders, Dockerfile) and you already built it, then instead run:

```
docker-compose up
```

When you want to shut down docker container
```
docker-compose up
```

## After the setup:

To start **airflow**, go to:
http://localhost:8090

The account created has the login airflow and the password airflow.


To start **mlflow**, go to:
http://localhost:5000 


# Docker

To set up my docker-compose I used the guide in the link below and modified it.
https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

Modifications:
- Created a Dockerfile i build instead of using the airflow-image. Then I can install my packages needed for my project. 
- Pointed out folders in volumes.
- Define AIRFLOW_UID to root user (0:0)
- added "PYTHONPATH: /opt/airflow/src" so it knows where to look for modules.
- Turned off loading example DAGs.
- added mlflow service in airflow-webserver


# Data 

Dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
