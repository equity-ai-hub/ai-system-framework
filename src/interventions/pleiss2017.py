import os

import numpy as np
import pandas as pd
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)
from aif360.datasets import StandardDataset

import src.interventions.utils as utils
from src.metrics import Metrics


def calibration_probabilities(labels, scores, n_bins=10):
    """
    Computes calibration probabilities per bin (i.e. P(Y = 1 | score)) for a
    set of scores and labels.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    probs = np.zeros(n_bins)

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == 1:
            # allow equality with one in the final bin
            high = 1.01
        mask = (scores >= low) & (scores < high)
        probs[i] = labels[mask].mean()
    return probs


def calibration_difference(labels, scores, attr, n_bins=10):
    """
    Computes average calibration difference between protected groups. Currently
    assumes binary protected attribute.
    """
    mask = attr == 1

    a0_calibration_probabilities = calibration_probabilities(labels[~mask], scores[~mask], n_bins)
    a1_calibration_probabilities = calibration_probabilities(labels[mask], scores[mask], n_bins)

    # if a bin is empty we get a nan, so use nanmean to aggregate only over
    # mutually non-empty bins
    return np.nanmean(np.abs(a0_calibration_probabilities - a1_calibration_probabilities))


def calibration(
    dataset_id: str,
    selected_model: str,
    local_data_path: str,
    sensitive_attr,
    privileged_groups,
    unprivileged_groups,
    cost_constraint: str = "weighted",
    num_folds: int = 10,
    path_to_save: str = None,
    **kwargs,
):
    """This intervention is based on the Pleiss et al. (2017) intervention, which is a
    post-processing technique that modifies the predictions of a model, relaxing the equalized odds
    to achieve calibration. The intervention is based on the CalibratedEqOddsPostprocessing algorithm
    from the AIF360 library.

    Args:
        dataset_id (str): The dataset id.
        selected_model (str): The selected model.
        local_data_path (str): The local data path.
        sensitive_attr (_type_): The name of the sensitive attribute.
        privileged_groups (_type_): The privileged groups.
        unprivileged_groups (_type_): The unprivileged groups.
        cost_constraint (str, optional): The cost constraint, to be used as calibrator.
            Options are "fnr", "fpr", "weighted". Defaults to "weighted".
        num_folds (int, optional): Number of folds available. Defaults to 10.
        path_to_save (str, optional): The path to save the results. Defaults to None.
    """

    # load the local dataset and the preprocessed data (without one-hot encoding)
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

    # run the intervention for each fold: 10 folds as 10 iterations
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

        # Predict the scores (calibration works with probabilites) for the validation and test set using the baseline model
        base_val_prob = baseline_model.predict_proba(val_oh.drop(["LABELS"], axis=1))[:, 1]
        val_standard_pred = val_standard.copy(deepcopy=True)
        val_standard_pred.scores = base_val_prob.reshape(-1, 1)

        base_test_prob = baseline_model.predict_proba(test_oh.drop("LABELS", axis=1))[:, 1]
        test_standard_pred = test_standard.copy(deepcopy=True)
        test_standard_pred.scores = base_test_prob.reshape(-1, 1)
        # baseline_test_pred = base_test_prob > 0.5

        # Define the binary values for the (un-)privileged groups
        # "SEX": {1.0: "Male", 2.0: "Female"},
        privileged_groups_oh = [{protected_attribute_names_oh: 1.0}]
        unprivileged_groups_oh = [{protected_attribute_names_oh: 0.0}]

        # Learn parameters to equal opportunity and apply to create a new dataset
        calibratedpp = CalibratedEqOddsPostprocessing(
            privileged_groups=privileged_groups_oh,
            unprivileged_groups=unprivileged_groups_oh,
            cost_constraint=cost_constraint,
            seed=42,
        )
        calibratedpp = calibratedpp.fit(val_standard, val_standard_pred)

        test_standard_pred_cpp = calibratedpp.predict(test_standard_pred)
        test_scores_cpp = test_standard_pred_cpp.scores.flatten()
        # test_pred = test_scores_cpp > 0.5

        intv_predictions[f"fold_{fold}"] = pd.Series(test_scores_cpp)

        # Calculate the performance scores using the AIF360 library
        metrics_scores[f"fold_{fold}"] = Metrics.metrics_scores_aif360(
            df=test_preproc,
            y_pred=test_standard_pred_cpp.labels,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        others = Metrics.calculate_metrics(y_true=test_preproc["LABELS"], y_pred=test_standard_pred_cpp.labels)
        metrics_scores[f"fold_{fold}"].update(others)

        metrics_scores[f"fold_{fold}"]["calibration_difference"] = calibration_difference(
            test_standard.labels.flatten(),
            test_scores_cpp,
            test_preproc["SEX"],
        )

        conditional_scores[f"fold_{fold}"] = Metrics.conditional_metrics_scores_aif360(
            df=test_preproc,
            y_pred=test_standard_pred_cpp.labels,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

    print("Path to save: ", path_to_save)
    if path_to_save is not None:
        eval_path = f"{path_to_save}/evaluation/pleiss2017/{dataset_id}/calib_{cost_constraint}"
    else:
        eval_path = f"{local_data_path}/evaluation/pleiss2017/{dataset_id}/calib_{cost_constraint}"

    print(f"Saving results to {eval_path}")
    os.makedirs(eval_path, exist_ok=True)

    intv_predictions.to_csv(f"{eval_path}/{selected_model}_sufficiency_predictions.csv", index=False, encoding="utf-8")
    np.save(f"{eval_path}/{selected_model}_scores_sufficiency.npy", metrics_scores)
    np.save(f"{eval_path}/{selected_model}_conditional_scores_sufficiency.npy", conditional_scores)
