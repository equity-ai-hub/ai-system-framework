import io
import os

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.metrics import MDSSClassificationMetric
from sklearn.metrics import balanced_accuracy_score

import src.interventions.utils as utils
from src.metrics import Metrics
from src.model import Model


def data_reweighing(
    dataset_id: str,
    selected_model: str,
    num_folds: int,
    local_data_path: str = None,
    **kwargs,
):
    """
    INDEPENDENCE NON-CRIMINATION CRITERIA: Kamiran and Calders intervention is to achieve demographic parity.

    Args:
        dataset_id (str): The dataset id.
        selected_model (str): The selected model.
        num_folds (int): The number of folds.
        local_data_path (str): The local data path.
        **kwargs: The keyword arguments.

    Returns:
        None
    """
    # DATA
    from src.acs_baseline.acs_pipeline_functions import load_dataset

    test_oh, test_preproc = load_dataset(dataset_id, "test", local_data_path)

    # Standardize the data to be used with AIF360 algorithms
    test_standard = StandardDataset(
        df=test_oh,
        label_name="LABELS",
        favorable_classes=[1.0],
        protected_attribute_names=["SEX_1.0"],
        privileged_classes=[[1.0]],
    )

    intv_predictions = pd.DataFrame()
    intv_predictions["y_true"] = pd.Series(list(test_standard.labels.flatten()))
    metrics_scores, conditional_scores = {}, {}

    for fold in range(num_folds):
        path = f"{local_data_path}/artifacts/acs_{dataset_id}/{selected_model}"
        train_oh, val_oh, _ = utils._get_model_and_data_local(selected_model, fold, path)

        # Standardize the data to be used with AIF360 algorithms
        train_standard = StandardDataset(
            df=train_oh,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=["SEX_1.0"],
            privileged_classes=[[1.0]],
        )

        val_standard = StandardDataset(
            df=val_oh,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=["SEX_1.0"],
            privileged_classes=[[1.0]],
        )

        # Define the binary values for the (un-)privileged groups
        # "SEX": {1.0: "Male", 2.0: "Female"},
        privileged_groups = [{"SEX_1.0": 1.0}]
        unprivileged_groups = [{"SEX_1.0": 0.0}]

        preprocessor = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        preprocessor.fit(train_standard)

        # Transform validation set - intervention
        val_standard_transform = preprocessor.transform(val_standard)

        X_val = val_standard_transform.features
        y_val = val_standard_transform.labels.flatten()

        # TODO: improve model train adding a option to not split the data and use the complete data to train
        # In this case, the model will be trained with the transformed data for each validation and training samples

        from xgboost import XGBClassifier

        model_fair = XGBClassifier()
        model_fair = model_fair.fit(X_val, y_val, sample_weight=val_standard_transform.instance_weights)

        test_standard_pred = test_standard.copy(deepcopy=True)
        X_test = test_standard_pred.features
        y_test = test_standard.labels

        test_pred = model_fair.predict(X_test)
        test_standard_pred.labels = test_pred.reshape(-1, 1)
        # test_probs = model_fair.predict_proba(X_test)[:, 1]
        # test_pred = test_probs > 0.5

        intv_predictions[f"fold_{fold}"] = pd.Series(test_pred)

        # Calculate the performance scores using the AIF360 library
        metrics_scores[f"fold_{fold}"] = Metrics.metrics_scores_aif360(
            df=test_preproc, y_pred=test_standard_pred.labels
        )
        conditional_scores[f"fold_{fold}"] = Metrics.conditional_metrics_scores_aif360(
            df=test_preproc, y_pred=test_standard_pred.labels
        )

    # Export the result as the same hardt2016 intervention
    eval_path = f"{local_data_path}/evaluation/kamiran_calders2012/{dataset_id}"
    os.makedirs(eval_path, exist_ok=True)

    intv_predictions.to_csv(f"{eval_path}/{selected_model}_independence_predictions.csv", index=False, encoding="utf-8")
    np.save(f"{eval_path}/{selected_model}_scores_independence.npy", metrics_scores)
    np.save(f"{eval_path}/{selected_model}_conditional_scores_independence.npy", conditional_scores)
