import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Metrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        return {
            "acc": accuracy_score(y_true, y_pred),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred),
        }

    @staticmethod
    def conditional_metrics_scores_aif360(df, y_pred, unprivileged_groups=None, privileged_groups=None):
        """Calculate the performance scores using the AIF360 library

        Args:
            df (pandas.DataFrame): The dataset.
            y_pred (numpy.ndarray): The predicted labels.
            unprivileged_groups (list): The unprivileged groups.
            privileged_groups (list): The privileged groups.

        Returns:
            dict: The performance scores. True positive rate, true negative rate,
            false positive rate, false negative rate, positive predictive value,
            negative predictive value, false discover rate, false omission rate,
            and accuracy (optionally conditioned).
        """
        # TODO: add privileged and unprivileged groups in the model initialization
        from aif360.datasets import StandardDataset
        from aif360.metrics import MDSSClassificationMetric

        # https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.MDSSClassificationMetric.html#aif360.metrics.MDSSClassificationMetric
        # check if df has the column called "y_pred", if yes dropped
        if "y_pred" in df.columns:
            df = df.drop(columns=["y_pred"])

        privileged_groups = [{"SEX": 1.0}]
        unprivileged_groups = [{"SEX": 2.0}]

        test_standard = StandardDataset(
            df=df,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=["SEX"],
            privileged_classes=[[1.0]],
        )

        test_standard_pred = test_standard.copy(deepcopy=True)
        test_standard_pred.labels = y_pred

        aif360_metrics = MDSSClassificationMetric(
            test_standard,
            test_standard_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        _aif360_metrics = {}

        # Alterando nomes das chaves no novo dicionário para evitar duplicações
        _unp = aif360_metrics.performance_measures(privileged=False)
        _temp_dict_unp = {}
        for key, value in _unp.items():
            temp = f"UNP_{key}"
            _temp_dict_unp[temp] = value

        _aif360_metrics.update(_temp_dict_unp)

        _priv = aif360_metrics.performance_measures(privileged=True)
        _temp_dict_priv = {}
        for key, value in _priv.items():
            temp = f"PRIV_{key}"
            _temp_dict_priv[temp] = value

        _aif360_metrics.update(_temp_dict_priv)

        return _aif360_metrics

    @staticmethod
    def metrics_scores_aif360(df, y_pred, unprivileged_groups=None, privileged_groups=None):
        """Calculate the performance scores using the AIF360 library

        Args:
            df (pandas.DataFrame): The dataset.
            y_pred (numpy.ndarray): The predicted labels.
            unprivileged_groups (list): The unprivileged groups.
            privileged_groups (list): The privileged groups.

        Returns:
            dict: The performance scores. True positive rate, true negative rate,
            false positive rate, false negative rate, positive predictive value,
            negative predictive value, false discover rate, false omission rate,
            and accuracy (optionally conditioned).
        """
        # TODO: add privileged and unprivileged groups in the model initialization
        from aif360.datasets import StandardDataset
        from aif360.metrics import MDSSClassificationMetric

        # https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.MDSSClassificationMetric.html#aif360.metrics.MDSSClassificationMetric
        # check if df has the column called "y_pred", if yes dropped
        if "y_pred" in df.columns:
            df = df.drop(columns=["y_pred"])

        privileged_groups = [{"SEX": 1.0}]
        unprivileged_groups = [{"SEX": 2.0}]

        test_standard = StandardDataset(
            df=df,
            label_name="LABELS",
            favorable_classes=[1.0],
            protected_attribute_names=["SEX"],
            privileged_classes=[[1.0]],
        )

        test_standard_pred = test_standard.copy(deepcopy=True)
        test_standard_pred.labels = y_pred

        aif360_metrics = MDSSClassificationMetric(
            test_standard,
            test_standard_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        _aif360_metrics = aif360_metrics.performance_measures()
        _aif360_metrics["BAL_ACC"] = balanced_accuracy_score(test_standard.labels, test_standard_pred.labels)

        _aif360_metrics["ABS_AVG_ODDS_DIFF"] = aif360_metrics.average_abs_odds_difference()
        _aif360_metrics["EQ_OPP_DIFF"] = aif360_metrics.equal_opportunity_difference()  # true positive rate difference
        _aif360_metrics["DI"] = aif360_metrics.disparate_impact()
        _aif360_metrics["STAT_PAR_DIFF"] = aif360_metrics.statistical_parity_difference()

        return _aif360_metrics

    @staticmethod
    def bootstrap_ci(data: list, alpha=0.95, n_bootstraps=10000):
        rng = np.random.default_rng(seed=42)  # For reproducibility
        bootstrapped_scores = np.array(
            [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)]
        )

        lower_bound = np.percentile(bootstrapped_scores, (1 - alpha) * 100 / 2)
        upper_bound = np.percentile(bootstrapped_scores, (1 + alpha) * 100 / 2)
        return lower_bound, upper_bound

    @staticmethod
    def cls_report_and_cm(y_test, y_pred):
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return cls_report, cm

    @staticmethod
    def get_train_stats(scores):
        """
        Calculate mean and confidence intervals for each metric
        """
        mean_scores = {k: np.mean(v) for k, v in scores.items()}
        confidence_intervals = {}

        confidence_intervals["accuracy"] = Metrics.bootstrap_ci(scores["accuracy"])
        confidence_intervals["f1"] = Metrics.bootstrap_ci(scores["f1"])
        confidence_intervals["precision"] = Metrics.bootstrap_ci(scores["precision"])
        confidence_intervals["recall"] = Metrics.bootstrap_ci(scores["recall"])

        return mean_scores, confidence_intervals

    @staticmethod
    def conf_interval(scores):
        mean = scores.mean()
        sem = stats.sem(scores)
        ci = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=sem)
        lower_ci = np.abs(mean - ci[0]).round(4)
        # upper_ci = np.abs(mean - ci[1]).round(4)
        ci = f"+/- {lower_ci}"
        return ci
