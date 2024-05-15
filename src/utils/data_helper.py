import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
ARTIFACTS_DIR = f"{ROOT_PATH}/artifacts/baseline/results"
DATA_DIR = f"{ROOT_PATH}/data/baseline/processed"


def split_dataset(df, test_size=0.2, random_state=42, stratify=None):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    return train, test


def one_hot_encode(df: pd.DataFrame, categorical_features: list):
    # Convert categorical features to string
    df[categorical_features] = df[categorical_features].astype(str)
    encode_df = pd.concat([df, pd.get_dummies(df.loc[:, categorical_features], dtype=np.int32)], axis=1)
    return encode_df


def clean_string(string: str):
    string = string.strip().lower().replace(" ", "_")
    string = string.strip().lower().replace("-_", "")
    string = string.strip().lower().replace("-", "_")
    return string


def get_metadata_scores(dataset_name: str, res_dir: str = None):
    df = pd.DataFrame(
        columns=[
            "model_name",
            "kfold",
            "acc",
            "bal_acc",
            "f1",
            "f1_macro",
            "precision",
            "recall",
            "roc_auc",
        ]
    )
    if res_dir is None:
        res_dir = "src/artifacts/results/"

    res_folder_path = os.path.join(res_dir, dataset_name)
    for folder_name in os.listdir(res_folder_path):
        model_folder = os.path.join(res_folder_path, folder_name)
        for kfolds in os.listdir(model_folder):
            kfold_path = os.path.join(model_folder, kfolds)
            for files in os.listdir(kfold_path):
                file_path = os.path.join(kfold_path, files)
                if file_path.endswith("_scores.npy"):
                    res = np.load(file_path, allow_pickle=True).item()
                    metrics_data = res["metrics"]
                    data = {
                        "model_name": res["model_name"],
                        "kfold": res["kfold"],
                        "sampling_method": res["sampling_method"],
                        **metrics_data,
                    }
                    df.loc[len(df)] = pd.Series(data)
    return df


def return_best_kfold(df, artifacts_path, choosed_model, dataset_name):
    sorted_df = df.groupby("model_name", group_keys=False).apply(
        lambda group: group.sort_values(["bal_acc"], ascending=False)
    )
    unique_best = sorted_df.drop_duplicates("model_name")
    for index, row in unique_best.iterrows():
        if row["model_name"] == choosed_model:
            model_dir = f"{artifacts_path}/{dataset_name}/{row['model_name']}/fold_{row['kfold']}/"
    return model_dir


def load_test_data(dataset_name: str):
    df_onehot = pd.read_csv(f"{DATA_DIR}/{dataset_name}/df-test-onehot.csv")
    df_preproc = pd.read_csv(f"{DATA_DIR}/{dataset_name}/df-test.csv")
    return df_onehot, df_preproc


def get_scores_from_test(
    dataset_name: str, dataset_target: str, res_dir: str = None, model_name: str = None, sensitive_attr: str = None
):
    import joblib
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
    )

    from src.metrics import Metrics

    test_oh, test = load_test_data(dataset_name)

    if res_dir is None:
        res_dir = "src/artifacts/results/"

    dfm = pd.DataFrame(
        columns=[
            "demo_parity_diff",
            "demo_parity_ratio",
            "eq_opp_diff",
            "eq_opp_ratio",
            "acc",
            "bal_acc",
            "f1",
            "f1_macro",
            "precision",
            "recall",
            "roc_auc",
        ]
    )
    res_folder_path = os.path.join(res_dir, dataset_name)
    for folder_name in os.listdir(res_folder_path):
        if folder_name == model_name:
            model_folder = os.path.join(res_folder_path, folder_name)
            for kfolds in os.listdir(model_folder):
                kfold_path = os.path.join(model_folder, kfolds)
                for file in os.listdir(kfold_path):
                    if file.endswith(".pkl"):
                        model_path = os.path.join(kfold_path, file)
                        model = joblib.load(model_path)

                        # Model accuracy and metrics on test set
                        test_prob = model.predict_proba(test_oh.drop(columns=[dataset_target]))[:, 1]
                        test_pred = test_prob > 0.5
                        test_metrics = Metrics.calculate_metrics(test_oh[dataset_target], test_pred)
                        data = {
                            "demo_parity_diff": demographic_parity_difference(
                                test_oh[dataset_target], test_pred, sensitive_features=test_oh[sensitive_attr]
                            ),
                            "demo_parity_ratio": demographic_parity_ratio(
                                test_oh[dataset_target], test_pred, sensitive_features=test_oh[sensitive_attr]
                            ),
                            "eq_opp_diff": equalized_odds_difference(
                                test_oh[dataset_target], test_pred, sensitive_features=test_oh[sensitive_attr]
                            ),
                            "eq_opp_ratio": equalized_odds_ratio(
                                test_oh[dataset_target], test_pred, sensitive_features=test_oh[sensitive_attr]
                            ),
                            **test_metrics,
                        }
                        dfm.loc[len(dfm)] = pd.Series(data)

    return dfm


def load_data_and_model(dataset_name, choosed_model, scores):
    def load_test_data(dataset_name: str):
        df_onehot = pd.read_csv(f"{DATA_DIR}/{dataset_name}/df-test-onehot.csv")
        df_preproc = pd.read_csv(f"{DATA_DIR}/{dataset_name}/df-test.csv")
        return df_onehot, df_preproc

    kfold_path = return_best_kfold(scores, ARTIFACTS_DIR, choosed_model, dataset_name)
    print(f"Best kfold path: {kfold_path}")

    for file in os.listdir(kfold_path):
        if file.endswith("train_preproc.csv"):
            csv_train = os.path.join(kfold_path, file)
        if file.endswith("train_oh.csv"):
            csv_train_oh = os.path.join(kfold_path, file)
        if file.endswith("_val_preproc.csv"):
            csv_val = os.path.join(kfold_path, file)
        if file.endswith("_val_oh.csv"):
            csv_val_oh = os.path.join(kfold_path, file)
        if file.endswith(".pkl"):
            model_path = os.path.join(kfold_path, file)
            model = joblib.load(model_path)

    train = pd.read_csv(csv_train)
    train_oh = pd.read_csv(csv_train_oh)

    val = pd.read_csv(csv_val)
    val_oh = pd.read_csv(csv_val_oh)
    val_oh = val_oh.drop(columns=["y_pred"])

    test_oh, test = load_test_data(dataset_name)

    return model, train, train_oh, val, val_oh, test, test_oh
