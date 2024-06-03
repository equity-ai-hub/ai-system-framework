FROM apache/airflow:slim-2.9.0-python3.9

COPY pyproject.toml .
COPY setup.py .

COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir apache-airflow[google,postgres]==${AIRFLOW_VERSION}

SHELL ["/bin/bash", "-o", "pipefail", "-e", "-u", "-x", "-c"]

USER 0
