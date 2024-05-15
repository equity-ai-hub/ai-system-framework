import io

import joblib
import pandas as pd

from src.gcs import CloudStorageData


def _get_model_and_data(dataset_id: str, selected_model: str, fold_num: int):

    artifacts = f"artifacts/{dataset_id}/{selected_model}"
    gcs = CloudStorageData(gcs_bucket="research-acs-data")

    model_filename = f"{artifacts}/fold_{fold_num}/{selected_model}.pkl"
    _model_obj = gcs.download_from_gcs(model_filename)
    model_pck = io.BytesIO(_model_obj)
    model_pck.seek(0)
    model = joblib.load(model_pck)

    val_filename = f"{artifacts}/fold_{fold_num}/{selected_model}_fold_{fold_num}_val_oh.csv"
    val = gcs.download_from_gcs(val_filename)

    train_filename = f"{artifacts}/fold_{fold_num}/{selected_model}_fold_{fold_num}_train_oh.csv"
    train = gcs.download_from_gcs(train_filename)

    df_val_oh = pd.read_csv(io.BytesIO(val))
    df_val_oh = df_val_oh.drop("y_pred", axis=1)

    df_train_oh = pd.read_csv(io.BytesIO(train))
    return df_train_oh, df_val_oh, model


def _get_model_and_data_local(selected_model: str, fold_num: int, local_data_path: str = None):

    model_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}.pkl"
    model = joblib.load(model_filename)

    val_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}_fold_{fold_num}_val_oh.csv"
    df_val_oh = pd.read_csv(val_filename)
    df_val_oh = df_val_oh.drop("y_pred", axis=1)

    train_filename = f"{local_data_path}/fold_{fold_num}/{selected_model}_fold_{fold_num}_train_oh.csv"
    df_train_oh = pd.read_csv(train_filename)

    return df_train_oh, df_val_oh, model
