import os

import numpy as np
import pandas as pd


def get_metadata_scores(dataset_name: str, res_dir: str = None):
    df_metrics = pd.DataFrame(
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
                    aeq_scores = res["aeq_scores"]
                    data = {
                        "model_name": res["model_name"],
                        "kfold": res["kfold"],
                        **metrics_data,
                    }
                    df_metrics.loc[len(df_metrics)] = pd.Series(data)
                    df_aeq = pd.DataFrame(aeq_scores)
    return df_metrics, df_aeq