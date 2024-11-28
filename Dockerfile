FROM apache/airflow:2.10.3-python3.9

# Install additional dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

USER root

RUN apt-get update && apt-get install -y git && apt-get clean
RUN mkdir -p /opt/airflow/tmp && chmod -R 777 /opt/airflow/tmp
RUN apt-get update && apt-get install -y libgomp1

USER airflow

# Set PYTHONPATH
ENV PYTHONPATH="/opt/airflow/src:${PYTHONPATH}"

