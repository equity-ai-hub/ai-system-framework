from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.metrics import Metrics


class Model:
    def __init__(
        self,
        model_name,
        dataset_id,
        df_one_hot,
        df_preproc,
        target,
        sensitive_attr,
        privileged_groups,
        unprivileged_groups,
    ):
        self.df_one_hot = df_one_hot
        self.df_preproc = df_preproc
        self.scores = None
        self.model = None
        self.evalution_scores = None
        self.target = target
        self.sensitive_attr = sensitive_attr
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.dataset_id = dataset_id  # identifier for the dataset
        self._initialize_model(model_name)

    def _initialize_model(self, model_name: str):
        models = {
            "logistic_regression": LogisticRegression(solver="liblinear", max_iter=1000),  # liblinear to COMPAS
            "mlp": MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, alpha=0.01, random_state=42),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
            "xgboost": XGBClassifier(),
            "decision_tree": DecisionTreeClassifier(),
        }

        if model_name in models:
            self.model = models[model_name]
        else:
            raise ValueError("Invalid model name")

    def train(
        self,
        n_folds: int = 10,
        data_dir: str = None,
        sampling_method: str = "no_sampling",
        sample_weight: Any = None,
    ):
        """Train the model using StratifiedKFold cross validation and export the model and training scores.
        Return the best model based on the average accuracy score.

        Args:
            n_folds (int, optional): _description_. Defaults to 5.
            data_dir (str, optional): _description_. Defaults to None.

        Returns:
            model: trained model
        """
        # get the original preprocessed dataset without one-hot encoding for folds data visualization
        X_preproc, y_preproc = (self.df_preproc.drop(self.target, axis=1), self.df_preproc[self.target])
        X, y = (self.df_one_hot.drop(self.target, axis=1), self.df_one_hot[self.target])

        valid_values = [
            "no_sampling",
            "undersampling",
            "oversampling",
        ]

        if sampling_method not in valid_values:
            raise ValueError("Invalid sampling method. Allowed values: 'no_sampling', 'undersampling', 'oversampling'")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        if self.model is None:
            raise ValueError("Model has not been initialized")

        for train_idx, val_idx in skf.split(X, y):
            n_folds = n_folds - 1
            X_fold_train, X_fold_val = (X.iloc[train_idx], X.iloc[val_idx])
            y_fold_train, y_fold_val = (y.iloc[train_idx], y.iloc[val_idx])

            X_preproc_fold_train, X_preproc_fold_val = (X_preproc.iloc[train_idx], X_preproc.iloc[val_idx])
            y_preproc_fold_train, y_preproc_fold_val = (y_preproc.iloc[train_idx], y_preproc.iloc[val_idx])

            if sampling_method == "oversampling":
                X_resampled, y_resampled = Model.smote_balance(X_fold_train, y_fold_train)
                self.model.fit(X_resampled, y_resampled, sample_weight=sample_weight)

            elif sampling_method == "undersampling":
                X_resampled, y_resampled = Model.random_undersample(X_fold_train, y_fold_train)
                self.model.fit(X_resampled, y_resampled, sample_weight=sample_weight)

            elif sampling_method == "no_sampling":
                if isinstance(self.model, MLPClassifier):
                    self.model.fit(X_fold_train, y_fold_train)
                else:
                    self.model.fit(X_fold_train, y_fold_train, sample_weight=sample_weight)

            y_fold_pred = self.model.predict(X_fold_val)
            try:
                y_fold_predict_proba = self.model.predict_proba(X_fold_val)[:, 1]
            except AttributeError:
                y_fold_predict_proba = np.zeros(len(y_fold_pred))
            y_pred = pd.Series(y_fold_pred, name="y_pred", index=X_fold_val.index)
            y_pred_proba = pd.Series(y_fold_predict_proba, name="y_pred_proba", index=X_fold_val.index)

            # save all models artifacts for each fold if data_dir is not None
            val_fold_preproc = pd.concat(
                [X_preproc_fold_val, y_preproc_fold_val, y_pred, y_pred_proba],
                axis=1,
            )
            val_fold_oh = pd.concat([X_fold_val, y_fold_val, y_pred], axis=1)

            train_fold_preproc = pd.concat(
                [X_preproc_fold_train, y_preproc_fold_train],
                axis=1,
            )
            train_fold_oh = pd.concat([X_fold_train, y_fold_train], axis=1)

            generic_metrics = Metrics.calculate_metrics(y_true=y_fold_val, y_pred=y_fold_pred)
            scores = Metrics.metrics_scores_aif360(
                df=val_fold_preproc,
                y_pred=y_fold_pred,
                sensitive_attr=self.sensitive_attr,
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
            )
            self.scores = {
                "model_name": self.model.__class__.__name__,
                "kfold": n_folds,
                "generic_metrics": generic_metrics,
                # "sampling_method": sampling_method,
                "scores": scores,
            }
            if data_dir is not None:
                print(f"Exporting model artifacts to {data_dir}")
                self._export_training_artifacts(
                    data_dir,
                    train_fold_oh,
                    train_fold_preproc,
                    val_fold_oh,
                    val_fold_preproc,
                    n_folds,
                )

    def _export_training_artifacts(
        self,
        data_dir: str,
        train_fold_oh: pd.DataFrame,
        train_fold_preproc: pd.DataFrame,
        val_fold_oh: pd.DataFrame,
        val_fold_preproc: pd.DataFrame,
        fold_num: int,
        # xtab: pd.DataFrame,
    ):
        model_name = self.model.__class__.__name__
        path_dir = f"{data_dir}/{model_name}/fold_{fold_num}"
        Path(path_dir).mkdir(parents=True, exist_ok=True)

        # save model
        joblib.dump(self.model, f"{path_dir}/{model_name}.pkl")

        # save scores, model scores and data fold
        np.save(f"{path_dir}/{model_name}_scores.npy", self.scores)

        p = f"{path_dir}/{model_name}_fold_{fold_num}"
        # TRAIN
        train_fold_oh.to_csv(f"{p}_train_oh.csv", index=False, encoding="utf-8")
        train_fold_preproc.to_csv(f"{p}_train_preproc.csv", index=False, encoding="utf-8")

        # VALIDATION
        val_fold_oh.to_csv(f"{p}_val_oh.csv", index=False, encoding="utf-8")
        val_fold_preproc.to_csv(f"{p}_val_preproc.csv", index=False, encoding="utf-8")

    @staticmethod
    def smote_balance(df: pd.DataFrame, target: str):
        """Balance the dataset using smote method
        Args:
            df (pd.DataFrame): dataset
            target (str, optional): target column.

        Returns:
            X, y: balanced dataset
        """
        oversample = SMOTE(random_state=42)
        X_res, y_res = oversample.fit_resample(df, target)
        return X_res, y_res

    @staticmethod
    def random_undersample(df: pd.DataFrame, target: str):
        """Balance the dataset using random undersampling
        Args:
            df (pd.DataFrame): dataset
            target (str, optional): target column.

        Returns:
            X, y: balanced dataset
        """
        rus = RandomUnderSampler(random_state=42)  # sampling_strategy=ratio
        X_res, y_res = rus.fit_resample(df, target)
        return X_res, y_res
