import io

import pandas as pd

from src.gcs import CloudStorageData


def load_dataset(type_data: str, dataset_id: str, **kwargs):
    gcs_data_path = kwargs["ti"].xcom_pull(
        dag_id="00_data_process", task_ids="process_data", key="return_value", include_prior_dates=True
    )
    print(gcs_data_path)

    gcs = CloudStorageData(gcs_bucket="research-acs-data")

    if type_data == "train":
        df_obj = gcs.download_from_gcs(f"{gcs_data_path}/acs_{dataset_id}_train.csv")
        df_oh_obj = gcs.download_from_gcs(f"{gcs_data_path}/acs_{dataset_id}_train_oh.csv")
    elif type_data == "test":
        df_obj = gcs.download_from_gcs(f"{gcs_data_path}/acs_{dataset_id}_test.csv")
        df_oh_obj = gcs.download_from_gcs(f"{gcs_data_path}/acs_{dataset_id}_test_oh.csv")

    # turn into data that can be serialized by airflow
    df = pd.read_csv(io.BytesIO(df_obj))
    df_oh = pd.read_csv(io.BytesIO(df_oh_obj))

    # push the bytes object to xcom
    kwargs["ti"].xcom_push(key=f"df_{dataset_id}_{type_data}", value=df)
    kwargs["ti"].xcom_push(key=f"df_{dataset_id}_{type_data}_oh", value=df_oh)
