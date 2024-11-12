import os
from typing import Any

import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset

import src.interventions.utils as utils
from src.algorithms.eq_odds_postprocessing import EqOddsPostprocessing
from src.metrics import Metrics


def threshold_modification(
    dataset_id: str,
    selected_model: str,
    local_data_path: str,
    sensitive_attr: Any,
    privileged_groups: Any,
    unprivileged_groups: Any,
    num_folds: int = 10,
    path_to_save: str = None,
    **kwargs,
):
    """
    Hardt intervention can be used to achieve equalized odds and satisfy the separation non-discrimination
    fairness.

    The EqOddsPostprocessing algorithm is a post-processing technique that optimizes over calibrated
    predictors to find probabilities with which to change output labels with an equalized odds objective.
    The algorithm is described in Hardt, Price, and Srebro," Equality of Opportunity in Supervised
    Learning" 2016. The implementation here is based on the one in the AI Fairness 360 toolkit.

    The EqOddsPostprocessing algorithm requires a base model that supports predict_proba() method.
    The base model is expected to be a binary classifier, and is used to predict the labels on the
    validation set. The EqOddsPostprocessing algorithm returns a new model that can be used to predict
    labels for the test set.

    Args:
        dataset_id (str): The dataset id.
        selected_model (str): The selected model.
        local_data_path (str): The local data path.
        sensitive_attr (Any): The sensitive attribute.
        privileged_groups (Any): The privileged groups.
        unprivileged_groups (Any): The unprivileged groups.
        num_folds (int): The number of folds.
        path_to_save (str): The path to save the results.
        **kwargs: The keyword arguments.

    Returns:
        None
    """
    # DATA
    from src.utils.acs_pipeline_functions import load_dataset

    test_oh, test_preproc = load_dataset(dataset_id, "test", local_data_path)
    protected_attribute_names_oh = f"{sensitive_attr}_1.0"

    # Standardize the data to be used with AIF360 algorithms
    test_standard = StandardDataset(
        df=test_oh,
        label_name="LABELS",
        favorable_classes=[1.0],
        protected_attribute_names=[protected_attribute_names_oh],
        privileged_classes=[[1.0]],
    )

    intv_predictions = pd.DataFrame()
    intv_predictions["y_true"] = pd.Series(list(test_standard.labels.flatten()))
    metrics_scores, conditional_scores = {}, {}

    # run the intervention for each fold - 10 iterations
    for fold in range(num_folds):
        path = f"{local_data_path}/artifacts/acs_{dataset_id}/{selected_model}"
        _, val_oh, baseline_model = utils._get_model_and_data_local(selected_model, fold, path)

        # Standardize the data to be used with AIF360 algorithms
        val_standard = StandardDataset(
            df=val_oh,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=[protected_attribute_names_oh],
            privileged_classes=[[1.0]],
        )

        # Predict the labels for the validation and test set using the baseline model
        base_val_pred = baseline_model.predict(val_oh.drop(["LABELS"], axis=1))
        val_standard_pred = val_standard.copy(deepcopy=True)
        val_standard_pred.labels = base_val_pred.reshape(-1, 1)

        base_test_pred = baseline_model.predict(test_oh.drop("LABELS", axis=1))
        test_standard_pred = test_standard.copy(deepcopy=True)
        test_standard_pred.labels = base_test_pred.reshape(-1, 1)

        # Define the binary values for the (un-)privileged groups
        # "SEX": {1.0: "Male", 2.0: "Female"},
        # "RACE": {1.0: "White", 2.0: "Black"},
        privileged_groups_oh = [{protected_attribute_names_oh: 1.0}]
        unprivileged_groups_oh = [{protected_attribute_names_oh: 0.0}]

        # Create the EqOddsPostprocessing object, fixed seed for reproducibility
        postprocessor = EqOddsPostprocessing(
            privileged_groups=privileged_groups_oh,
            unprivileged_groups=unprivileged_groups_oh,
            seed=42,
        )

        # eopp fixing error cast labels
        val_standard.labels = val_standard.labels.astype(int)
        val_standard_pred.labels = val_standard_pred.labels.astype(int)

        postprocessor = postprocessor.fit(val_standard, val_standard_pred)
        test_standard_pred_pp = postprocessor.predict(test_standard_pred)
        test_scores_eopp = test_standard_pred_pp.labels.flatten().tolist()

        intv_predictions[f"fold_{fold}"] = pd.Series(test_scores_eopp)

        # Calculate the performance scores using the AIF360 library, save results per fold
        metrics_scores[f"fold_{fold}"] = Metrics.metrics_scores_aif360(
            df=test_preproc,
            y_pred=test_standard_pred_pp.labels,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        others = Metrics.calculate_metrics(y_true=test_preproc["LABELS"], y_pred=test_standard_pred_pp.labels)
        metrics_scores[f"fold_{fold}"].update(others)

        conditional_scores[f"fold_{fold}"] = Metrics.conditional_metrics_scores_aif360(
            df=test_preproc,
            y_pred=test_standard_pred_pp.labels,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

    if path_to_save is not None:
        eval_path = f"{path_to_save}/evaluation/hardt2016/{dataset_id}"
    else:
        eval_path = f"{local_data_path}/evaluation/hardt2016/{dataset_id}"

    os.makedirs(eval_path, exist_ok=True)

    intv_predictions.to_csv(f"{eval_path}/{selected_model}_separation_predictions.csv", index=False, encoding="utf-8")
    np.save(f"{eval_path}/{selected_model}_scores_separation.npy", metrics_scores)
    np.save(f"{eval_path}/{selected_model}_conditional_scores_separation.npy", conditional_scores)
