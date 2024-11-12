import os

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from fairlearn.metrics import demographic_parity_difference
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import src.interventions.utils as utils
from src.metrics import Metrics


def get_model_to_retrain(model_name: str):
    models = {
        "LogisticRegression": LogisticRegression(solver="liblinear", max_iter=1000),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, alpha=0.01, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(),
        "SVC": SVC(),
        "XGBClassifier": XGBClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
    }

    if model_name in models:
        return models[model_name]
    else:
        raise ValueError("Invalid model name")


def data_reweighing(
    dataset_id: str,
    selected_model: str,
    local_data_path: str,
    sensitive_attr,
    privileged_groups,
    unprivileged_groups,
    num_folds: int = 10,
    path_to_save: str = None,
    **kwargs,
):
    """
    Kamiran and Calders intervention is to achieve statistical (or demographic) parity and satisfy the
    independence non-discrimination fairness.

    The intervention is based on the reweighing algorithm, which is a preprocessing technique that modifies
    the training data to remove bias in the training data.


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

    intv_probabilities = pd.DataFrame()
    intv_probabilities["y_true"] = pd.Series(list(test_standard.labels.flatten()))

    metrics_scores, conditional_scores = {}, {}
    dpd_baseline, dpd_reweighing = {}, {}

    for fold in range(num_folds):
        path = f"{local_data_path}/artifacts/acs_{dataset_id}/{selected_model}"
        train_oh, val_oh, baseline_model = utils._get_model_and_data_local(selected_model, fold, path)

        # Standardize the data to be used with AIF360 algorithms
        train_standard = StandardDataset(
            df=train_oh,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=[protected_attribute_names_oh],
            privileged_classes=[[1.0]],
        )

        val_standard = StandardDataset(
            df=val_oh,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=[protected_attribute_names_oh],
            privileged_classes=[[1.0]],
        )

        # Define the binary values for the (un-)privileged groups
        # "SEX": {1.0: "Male", 2.0: "Female"},
        privileged_groups_oh = [{protected_attribute_names_oh: 1.0}]
        unprivileged_groups_oh = [{protected_attribute_names_oh: 0.0}]

        preprocessor = Reweighing(
            unprivileged_groups=unprivileged_groups_oh,
            privileged_groups=privileged_groups_oh,
        )
        preprocessor.fit(train_standard)

        # Transform validation set - intervention
        val_standard_transform = preprocessor.transform(val_standard)

        X_val = val_standard_transform.features
        y_val = val_standard_transform.labels.flatten()

        # TODO: improve model train adding a option to not split the data and use the complete data to train
        # In this case, the model will be trained with the transformed data for each validation and training samples

        model_fair = get_model_to_retrain(selected_model)
        model_fair = model_fair.fit(X_val, y_val, sample_weight=val_standard_transform.instance_weights)

        test_standard_pred = test_standard.copy(deepcopy=True)
        X_test = test_standard_pred.features
        # y_test = test_standard.labels

        test_pred = model_fair.predict(X_test)
        test_standard_pred.labels = test_pred.reshape(-1, 1)
        test_probs = model_fair.predict_proba(X_test)[:, 1]
        # test_pred = test_probs > 0.5

        intv_predictions[f"fold_{fold}"] = pd.Series(test_pred)
        intv_probabilities[f"fold_{fold}"] = pd.Series(test_probs)

        # Calculate the performance scores using the AIF360 library
        metrics_scores[f"fold_{fold}"] = Metrics.metrics_scores_aif360(
            df=test_preproc,
            y_pred=test_standard_pred.labels,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        others = Metrics.calculate_metrics(y_true=test_preproc["LABELS"], y_pred=test_standard_pred.labels)
        metrics_scores[f"fold_{fold}"].update(others)

        conditional_scores[f"fold_{fold}"] = Metrics.conditional_metrics_scores_aif360(
            df=test_preproc,
            y_pred=test_standard_pred.labels,
            sensitive_attr=sensitive_attr,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        # ============================================
        bl_test_probs = baseline_model.predict_proba(X_test)[:, 1]
        bl_test_pred = bl_test_probs > 0.5

        bl_dpd = demographic_parity_difference(
            test_preproc["LABELS"],
            bl_test_pred,
            sensitive_features=test_oh[protected_attribute_names_oh],
        )
        dpd = demographic_parity_difference(
            test_preproc["LABELS"],
            test_pred,
            sensitive_features=test_oh[protected_attribute_names_oh],
        )
        dpd_baseline[f"fold_{fold}"] = bl_dpd
        dpd_reweighing[f"fold_{fold}"] = dpd
        print(f"Baseline demographic parity difference: {bl_dpd:.3f}")
        print(f"Model demographic parity difference: {dpd:.3f}")

        # bl_acc = accuracy(test_preproc["LABELS"], bl_test_probs)
        # bl_dpd = demographic_parity_difference(
        #     test_preproc["LABELS"],
        #     bl_test_pred,
        #     sensitive_features=test.race_white,
        # )

        # acc = accuracy(test_preproc["LABELS"], test_probs)

        # print(f"Baseline model accuracy: {bl_acc:.3f}")
        # print(f"Model accuracy: {acc:.3f}")

    # Export the result as the same hardt2016 intervention
    if path_to_save is not None:
        eval_path = f"{path_to_save}/evaluation/kamiran_calders2012/{dataset_id}"
    else:
        eval_path = f"{local_data_path}/evaluation/kamiran_calders2012/{dataset_id}"

    os.makedirs(eval_path, exist_ok=True)

    intv_predictions.to_csv(f"{eval_path}/{selected_model}_independence_predictions.csv", index=False, encoding="utf-8")
    intv_probabilities.to_csv(
        f"{eval_path}/{selected_model}_independence_probabilities.csv", index=False, encoding="utf-8"
    )
    np.save(f"{eval_path}/{selected_model}_scores_independence.npy", metrics_scores)
    np.save(f"{eval_path}/{selected_model}_conditional_scores_independence.npy", conditional_scores)
    np.save(f"{eval_path}/{selected_model}_dpd_baseline.npy", dpd_baseline)
    np.save(f"{eval_path}/{selected_model}_dpd_reweighing.npy", dpd_reweighing)
