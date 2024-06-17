# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=no-member

"""
Module: nested

This module provides tools for nested cross validation.
"""

import json
import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

N_CPU = 4
SCORING_METRIC = "roc_auc_ovr"
with open(
    f"{os.path.dirname(os.path.realpath(__file__))}/param_distributions.json",
    "r",
    encoding="utf-8",
) as f:
    PARAM_DIST = json.load(f)["XGB"]


def run(
    X: pd.DataFrame,
    y: pd.DataFrame,
    folds: int = 5,
    seed: int = 8,
    n_splits: int = 1,
    outfn: str = "performance.csv",
):
    """
    Run nested cross validation.

    Parameters:
    - X: input data
    - y: integer target labels
    - folds: number of folds for the cross validation (for both layers)
    - seed: random state for splits and model initialization
    - n_splits: number of CV splits to average across overall
    """
    X, y = align(X, y)
    # Create empty dataframe to record performance
    performance = pd.DataFrame(
        index=[f"Fold {k}" for k in np.arange(folds)] + ["Average"],
        columns=[f"CV ROC {l}" for l in np.arange(n_splits)]
        + [f"Test ROC {l}" for l in np.arange(n_splits)]
        + [f"Test Precision {l}" for l in np.arange(n_splits)]
        + [f"Test Recall {l}" for l in np.arange(n_splits)]
        + [f"Test F1 {l}" for l in np.arange(n_splits)],
    )
    # Create empty dataframe to record feature importance
    f_imp = pd.DataFrame(
        0.0,
        index=[
            f"split {split} fold {fold}"
            for fold in range(folds)
            for split in range(n_splits)
        ],
        columns=X.columns,
    )
    # Create dataframe to record sample-wise performance (find hard samples)
    sample_correct = pd.DataFrame(
        0.0,
        index=X.index,
        columns=[f"split {split}" for split in range(n_splits)],
        dtype=float,
    )
    # Go over splits
    for split in range(n_splits):
        print(f"Running split {split+1}/{n_splits}...")
        # First layer of split
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + split)
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            # Define training set and test set
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Run grid search on layer 1 training set (second layer of split)
            gsCV = gs_hparam(X_train, y_train, seed=seed)
            performance = record_performance(
                gsCV, X_test, y_test, performance, fold=fold, split=split
            )
            correct_samples = find_correct_samples(gsCV, X_test, y_test)
            sample_correct.loc[correct_samples, f"split {split}"] = 1.0
            f_imp.loc[
                f"split {split} fold {fold}", gsCV.feature_names_in_
            ] = gsCV.best_estimator_["classifier"].feature_importances_
            # Calculate and record SHAP values
            if split == 0:  # only do it for the first split for now
                explainer = shap.Explainer(gsCV.best_estimator_.predict, X_train)
                shap_fold = explainer(X_test, max_evals=515)
                if fold == 0:
                    shap_all = shap.Explanation(
                        values=shap_fold.values,
                        base_values=shap_fold.base_values,
                        data=shap_fold.data,
                        feature_names=shap_fold.feature_names,
                        instance_names=list(X_test.index),
                    )
                else:
                    shap_all = shap.Explanation(
                        values=np.concatenate(
                            [shap_all.values, shap_fold.values], axis=0
                        ),
                        base_values=np.concatenate(
                            [shap_all.base_values, shap_fold.base_values], axis=0
                        ),
                        data=np.concatenate([shap_all.data, shap_fold.data], axis=0),
                        feature_names=shap_all.feature_names,
                        instance_names=list(shap_all.instance_names)
                        + list(X_test.index),
                    )
                with open(outfn.rsplit(".", maxsplit=1)[0] + "_shap.p", "wb") as f:
                    pickle.dump(shap_all, f)
    # Calculate average across folds
    performance.loc["Average", :] = (
        performance.loc[[f"Fold {l}" for l in np.arange(folds)], :]
        .mean(axis=0)
        .values.ravel()
    )
    # Save performance and feature importance scores
    performance.to_csv(outfn)
    f_imp.to_csv(outfn.rsplit(".", maxsplit=1)[0] + "_feature_importances.csv")
    sample_correct.to_csv(outfn.rsplit(".", maxsplit=1)[0] + "_samplewise_pf.csv")


def align(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align input and target by index values.

    Parameters:
        - Input
        - Target

    Returns:
        - Aligned input
        - Aligned target
    """
    # Align input and target dataframes by index
    overlap = list(set(X.index).intersection(y.index))
    X = X.loc[overlap, :]
    y = y.loc[overlap]
    return (X, y)


def gs_hparam(X: pd.DataFrame, y: pd.DataFrame, seed: int, scale=True) -> GridSearchCV:
    """
    Run GridSearch with default model (XGB) and defined hyperparameters.

    Parameters:
        X: input.
        y: target.
        seed: random state.
        scale: whether to add scaler to the Pipeline

    Returns:
        The post-gridsearch GridSearchCV object.
    """
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 30))
        clf = xgb.XGBClassifier(
            objective="reg:logistic",
            subsample=1,
            reg_alpha=0,
            reg_lambda=1,
            n_estimators=300,
            seed=seed,
            scale_pos_weight=y.sum().item() / len(y),
        )
        pipe = Pipeline([("scaler", scaler), ("classifier", clf)])
    else:
        clf = xgb.XGBClassifier(
            objective="reg:logistic",
            subsample=1,
            reg_alpha=0,
            reg_lambda=1,
            n_estimators=300,
            seed=seed,
            scale_pos_weight=y.sum().item() / len(y),
        )
        pipe = Pipeline([("classifier", clf)])
    gsCV = GridSearchCV(
        pipe,
        param_grid=PARAM_DIST,
        n_jobs=N_CPU,
        scoring=SCORING_METRIC,
        refit=True,
        cv=StratifiedKFold(n_splits=5, random_state=seed + 1, shuffle=True),
    )
    gsCV.fit(X, y)
    return gsCV


def record_performance(
    gsCV: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    performance: pd.DataFrame,
    fold: int,
    split: int,
) -> pd.DataFrame:
    """
    Record performance of the grid search process.
    """
    # First record CV performance
    _, opt_mean_score = find_optimal(gsCV)
    performance.loc[f"Fold {fold}", f"CV ROC {split}"] = opt_mean_score
    # Next record test set performance
    y_pred = gsCV.predict(X_test)
    y_prob = gsCV.predict_proba(X_test)
    if y_prob.shape[1] == 2:  # if binary class
        y_prob = y_prob[:, 1]
    test_roc = metrics.roc_auc_score(y_test, y_prob, multi_class="ovr")
    test_prec = metrics.precision_score(y_test, y_pred)
    test_recall = metrics.recall_score(y_test, y_pred)
    test_f1 = metrics.f1_score(y_test, y_pred)
    performance.loc[f"Fold {fold}", f"Test ROC {split}"] = test_roc
    performance.loc[f"Fold {fold}", f"Test Precision {split}"] = test_prec
    performance.loc[f"Fold {fold}", f"Test Recall {split}"] = test_recall
    performance.loc[f"Fold {fold}", f"Test F1 {split}"] = test_f1
    return performance


def find_correct_samples(
    gsCV: GridSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> list[int]:
    """
    Get the sample IDs that were correctly predicted.
    """
    y_pred = gsCV.predict(X_test)
    correct = list(X_test.iloc[y_test.values == y_pred, :].index)
    return correct


def find_optimal(gsobj: GridSearchCV) -> Tuple[dict, float]:
    opt_params = gsobj.best_params_
    opt_mean_score = np.mean(
        gsobj.cv_results_["mean_test_score"][
            np.all(
                [
                    (gsobj.cv_results_[f"param_{param_name}"] == param_vals)
                    for param_name, param_vals in opt_params.items()
                ],
                axis=0,
            )
        ]
    )
    return (opt_params, opt_mean_score)


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:  # no arguments
        iris_X, iris_y = load_iris(return_X_y=True, as_frame=True)
        run(iris_X, iris_y, n_splits=2)
    else:
        custom_X = pd.read_csv(args[1], index_col=0, header=0)
        custom_y = pd.read_csv(args[2], index_col=0, header=0).iloc[:, 0]
        custom_outfn = args[3]
        N_CPU = int(args[4])
        run(custom_X, custom_y, n_splits=5, outfn=custom_outfn)
