""" Get data from Folktable and process it for ACS baseline model."""

import os
import warnings

import pandas as pd

from src.acs import ACSDataset
from src.utils.data_utils import export_csv, save_obj_to_csv_file

warnings.filterwarnings("ignore")


def download_acs_data(acs_task: str, data_path: str = None):
    acs = ACSDataset()
    features = acs.get_data(download=True, task_name=acs_task)

    # local path to save raw data
    PATH_RAW_DATA = f"{data_path}/acs_{acs_task}/raw"
    os.makedirs(PATH_RAW_DATA, exist_ok=True)

    print(f"Saving raw data to {PATH_RAW_DATA}")
    save_obj_to_csv_file(features, PATH_RAW_DATA, f"acs_{acs_task}.csv")


def process_acs_data(acs_task: str = "employment", data_path: str = None):
    """Read raw data, split it into train and test sets, and save them in the processed folder.

    Args:
        acs_task (str, optional): ACS task to be processed. Defaults to "employment".
        data_storage (str, optional): Storage type. Defaults to "local".
        PATH_RAW_DATA (str, optional): Path to raw data. Defaults to None.

    Raises:
        ValueError: [description]
    """
    PATH_RAW_DATA = f"{data_path}/acs_{acs_task}/raw"
    # raise ValueError("Path to raw data is missing")
    df_acs = pd.read_csv(f"{PATH_RAW_DATA}/acs_{acs_task}.csv")

    acs = ACSDataset()
    train_csv, test_csv = acs.split_data(df=df_acs)

    PATH_TO_PROCESS_DATA = f"{data_path}/acs_{acs_task}/processed"
    os.makedirs(PATH_TO_PROCESS_DATA, exist_ok=True)

    # save dataframe splitted data locally
    export_csv(train_csv, PATH_TO_PROCESS_DATA, f"acs_{acs_task}_train.csv")
    export_csv(test_csv, PATH_TO_PROCESS_DATA, f"acs_{acs_task}_test.csv")

    # if return a csv bytes data
    # save_obj_to_csv_file(train_csv, PATH_TO_PROCESS_DATA, f"acs_{acs_task}_train.csv")
    # save_obj_to_csv_file(test_csv, PATH_TO_PROCESS_DATA, f"acs_{acs_task}_test.csv")
    # _train = pd.read_csv(io.BytesIO(train_csv))
    # _test = pd.read_csv(io.BytesIO(test_csv))
    # train_oh_csv = acs.preprocess_data(df=_train, dtype="csv")
    # test_oh_csv = acs.preprocess_data(df=_test, dtype="csv")

    # fmt: off
    if acs_task == "employment":
        cat_features = ["MAR", "DIS", "CIT", "MIG", "MIL", "ANC",
                        "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RACE"]
    elif acs_task == "income":
        cat_features = ["COW", "SCHL", "MAR", "SEX", "RACE"]
    elif acs_task == "public_coverage":
        cat_features = ["SCHL", "MAR", "SEX", "DIS", "CIT", "MIG", "MIL",
                        "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "ESR",
                        "ST", "FER", "RACE"]

    # fmt: on
    train_oh_csv = acs.preprocess_data(df=train_csv, categorical_features=cat_features, dtype="csv")
    test_oh_csv = acs.preprocess_data(df=test_csv, categorical_features=cat_features, dtype="csv")

    # save processed data locally
    print(f"Saving processed data to {PATH_TO_PROCESS_DATA}")

    save_obj_to_csv_file(train_oh_csv, PATH_TO_PROCESS_DATA, f"acs_{acs_task}_train_oh.csv")
    save_obj_to_csv_file(test_oh_csv, PATH_TO_PROCESS_DATA, f"acs_{acs_task}_test_oh.csv")


def load_dataset(acs_type, type_data, local_path_file=None):
    if type_data == "train":
        df = pd.read_csv(f"{local_path_file}/acs_{acs_type}/processed/acs_{acs_type}_train.csv")
        df_oh = pd.read_csv(f"{local_path_file}/acs_{acs_type}/processed/acs_{acs_type}_train_oh.csv")
    elif type_data == "test":
        df = pd.read_csv(f"{local_path_file}/acs_{acs_type}/processed/acs_{acs_type}_test.csv")
        df_oh = pd.read_csv(f"{local_path_file}/acs_{acs_type}/processed/acs_{acs_type}_test_oh.csv")

    return df_oh, df


def run_model_train(
    selected_model,
    acs_type,
    type_data,
    data_path,
    sensitive_attr="SEX",
    privileged_groups=[{"SEX": 1.0}],
    unprivileged_groups=[{"SEX": 2.0}],
    num_folds=10,
):
    from src.model import Model

    df_oh, df = load_dataset(acs_type=acs_type, type_data=type_data, local_path_file=data_path)

    model = Model(
        model_name=selected_model,
        df_one_hot=df_oh,
        df_preproc=df,
        target="LABELS",
        dataset_id=acs_type,
        sensitive_attr=sensitive_attr,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
    )
    path_to_save = f"{data_path}/artifacts/acs_{acs_type}/"
    print(f"Path to save: {path_to_save}")
    model.train(n_folds=num_folds, data_dir=path_to_save)


def models_evaluation(
    selected_model: str,
    dataset_id: str,
    data_path: str,
    sensitive_attr: str,
    privileged_groups: list,
    unprivileged_groups: list,
    num_folds: int = 10,
):
    import joblib
    import numpy as np

    from src.metrics import Metrics

    df_test_onehot, df_test_preproc = load_dataset(acs_type=dataset_id, type_data="test", local_path_file=data_path)
    artifacts = f"{data_path}/artifacts/acs_{dataset_id}/{selected_model}"
    os.makedirs(artifacts, exist_ok=True)
    # structure to save the predictions and scores for each fold
    # df = pd.DataFrame(columns=["model_name", "kfold", "y_true", "y_pred", "y_pred_proba"])

    base_predictions = pd.DataFrame()
    base_predictions["y_true"] = pd.Series(list(df_test_onehot["LABELS"]))

    base_probabilities = pd.DataFrame()
    base_probabilities["y_true"] = pd.Series(list(df_test_onehot["LABELS"]))

    scores, conditional_scores = {}, {}

    for fold in range(num_folds):
        model_pck = f"{artifacts}/fold_{fold}/{selected_model}.pkl"
        model = joblib.load(model_pck)

        y_true = df_test_onehot["LABELS"]
        y_fold_pred = model.predict(df_test_onehot.drop(columns=["LABELS"]))
        y_fold_pred_prob = model.predict_proba(df_test_onehot.drop(columns=["LABELS"]))[:, 1]
        # test_pred = test_prob > 0.5

        y_pred = pd.Series(y_fold_pred, name="y_pred", index=df_test_onehot.index)
        y_pred_prob = pd.Series(y_fold_pred_prob, name="y_pred_proba", index=df_test_onehot.index)

        # df_pred_and_probs = pd.concat([df_test_preproc, y_pred, y_pred_prob], axis=1)

        base_predictions[f"fold_{fold}"] = pd.Series(y_pred)
        base_probabilities[f"fold_{fold}"] = pd.Series(y_pred_prob)

        scores[f"fold_{fold}"] = Metrics.metrics_scores_aif360(
            df=df_test_preproc,
            y_pred=y_pred,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        others = Metrics.calculate_metrics(y_true=y_true, y_pred=y_pred)
        scores[f"fold_{fold}"].update(others)

        conditional_scores[f"fold_{fold}"] = Metrics.conditional_metrics_scores_aif360(
            df=df_test_preproc,
            y_pred=y_pred,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

    eval_path = f"{data_path}/evaluation/baseline/{dataset_id}"
    os.makedirs(eval_path, exist_ok=True)
    np.save(f"{eval_path}/{selected_model}_scores.npy", scores)
    np.save(f"{eval_path}/{selected_model}_conditional_scores.npy", conditional_scores)
    base_predictions.to_csv(f"{eval_path}/{selected_model}_predictions.csv", index=False, encoding="utf-8")
    base_probabilities.to_csv(f"{eval_path}/{selected_model}_probabilities.csv", index=False, encoding="utf-8")
