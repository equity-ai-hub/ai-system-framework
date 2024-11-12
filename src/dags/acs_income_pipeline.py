from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore

from src.utils.acs_pipeline_functions import (
    download_acs_data,
    models_evaluation,
    process_acs_data,
    run_model_train,
)
from src.interventions.pleiss2017 import calibration
from src.interventions.hardt2016 import threshold_modification
from src.interventions.kamiran_calders2012 import data_reweighing

ROOT_PATH = str(Path(__file__).parent.parent)
DATA_PATH = f"{ROOT_PATH}/src/data"

default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "start_date": datetime.today().strftime("%Y-%m-%d"),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": None,
}

with DAG(
    "income_task_pipeline",
    description="ML Pipeline: ACSIncome Task",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["local_storage"],
    max_active_tasks=7,
) as dag:
    download = PythonOperator(
        task_id="data_download",
        python_callable=download_acs_data,
        op_args=["income", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    process = PythonOperator(
        task_id="data_process",
        python_callable=process_acs_data,
        op_args=["income", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    train_logreg = PythonOperator(
        task_id="train_logreg",
        python_callable=run_model_train,
        op_args=["logistic_regression", "income", "train", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    train_random_forest = PythonOperator(
        task_id="train_random_forest",
        python_callable=run_model_train,
        op_args=["random_forest", "income", "train", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    train_xgboost = PythonOperator(
        task_id="train_xgboost",
        python_callable=run_model_train,
        op_args=["xgboost", "income", "train", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    train_dec_tree = PythonOperator(
        task_id="train_dec_tree",
        python_callable=run_model_train,
        op_args=["decision_tree", "income", "train", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    train_mlp = PythonOperator(
        task_id="train_mlp",
        python_callable=run_model_train,
        op_args=["mlp", "income", "train", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    baseline_model_eval = PythonOperator(
        task_id="baseline_model_eval",
        python_callable=models_evaluation,
        op_args=["XGBClassifier", "income", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    separation_intv_eval = PythonOperator(
        task_id="separation_intv_model_eval",
        python_callable=threshold_modification,
        op_args=["income", "XGBClassifier", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    indenpendence_intv_eval = PythonOperator(
        task_id="independence_intv_model_eval",
        python_callable=data_reweighing,
        op_args=["income", "XGBClassifier", DATA_PATH],
        show_return_value_in_logs=True,
        dag=dag,
    )

    sufficiency_intv_eval = PythonOperator(
        task_id="sufficiency_intv_model_eval",
        python_callable=calibration,
        op_args=["income", "XGBClassifier", DATA_PATH, "weighted"],
        show_return_value_in_logs=True,
        dag=dag,
    )

    (
        download >> process >> [train_logreg, train_random_forest],
        process >> [train_xgboost, train_dec_tree, train_mlp],
        train_xgboost
        >> baseline_model_eval
        >> [
            separation_intv_eval,
            indenpendence_intv_eval,
            sufficiency_intv_eval,
        ],
    )
