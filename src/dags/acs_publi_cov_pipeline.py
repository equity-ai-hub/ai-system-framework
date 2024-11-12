from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore

from src.interventions.hardt2016 import threshold_modification
from src.interventions.kamiran_calders2012 import data_reweighing
from src.interventions.pleiss2017 import calibration
from src.utils.acs_pipeline_functions import (
    download_acs_data,
    models_evaluation,
    process_acs_data,
    run_model_train,
)

ROOT_PATH = str(Path(__file__).parent.parent)
DATA_PATH = f"{ROOT_PATH}/src/data"

sensitive_attr = "RACE"
privileged_groups = [{"RACE": 1.0}]
unprivileged_groups = [{"RACE": 2.0}]

default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "start_date": datetime.today().strftime("%Y-%m-%d"),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": None,
}

with DAG(
    "public_coverage_task_pipeline",
    description="ML Pipeline: ACSPublic Coverage Task",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["local_storage"],
    max_active_tasks=7,
) as dag:
    download = PythonOperator(
        task_id="data_download",
        python_callable=download_acs_data,
        op_args=["public_coverage", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    process = PythonOperator(
        task_id="data_process",
        python_callable=process_acs_data,
        op_args=["public_coverage", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    train_logreg = PythonOperator(
        task_id="train_logreg",
        python_callable=run_model_train,
        op_args=[
            "logistic_regression",
            "public_coverage",
            "train",
            DATA_PATH,
            sensitive_attr,
            privileged_groups,
            unprivileged_groups,
        ],
        show_return_value_in_logs=True,
        dag=dag,
    )

    baseline_model_eval = PythonOperator(
        task_id="baseline_model_eval",
        python_callable=models_evaluation,
        op_args=[
            "LogisticRegression",
            "public_coverage",
            DATA_PATH,
            sensitive_attr,
            privileged_groups,
            unprivileged_groups,
        ],
        show_return_value_in_logs=True,
        dag=dag,
    )

    separation_intv_eval = PythonOperator(
        task_id="separation_intv_model_eval",
        python_callable=threshold_modification,
        op_args=[
            "public_coverage",
            "LogisticRegression",
            DATA_PATH,
            sensitive_attr,
            privileged_groups,
            unprivileged_groups,
        ],
        show_return_value_in_logs=True,
        dag=dag,
    )

    indenpendence_intv_eval = PythonOperator(
        task_id="independence_intv_model_eval",
        python_callable=data_reweighing,
        op_args=[
            "public_coverage",
            "LogisticRegression",
            DATA_PATH,
            sensitive_attr,
            privileged_groups,
            unprivileged_groups,
        ],
        show_return_value_in_logs=True,
        dag=dag,
    )

    sufficiency_intv_eval = PythonOperator(
        task_id="sufficiency_intv_model_eval",
        python_callable=calibration,
        op_args=[
            "public_coverage",
            "LogisticRegression",
            DATA_PATH,
            sensitive_attr,
            privileged_groups,
            unprivileged_groups,
        ],
        show_return_value_in_logs=True,
        dag=dag,
    )

    (
        download
        >> process
        >> train_logreg
        >> baseline_model_eval
        >> [
            separation_intv_eval,
            indenpendence_intv_eval,
            sufficiency_intv_eval,
        ],
    )
