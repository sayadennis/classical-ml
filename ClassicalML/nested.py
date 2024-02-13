# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Module: nested

This module provides tools for nested cross validation.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    # Align input and target dataframes by index
    overlap = list(set(X.index).intersection(y.index))
    X = X.loc[overlap, :]
    y = y.loc[overlap]
    # Create empty dataframe to record performance
    performance = pd.DataFrame(
        index=[f"Fold {k}" for k in np.arange(folds)] + ["Average"],
        columns=[f"CV ROC {l}" for l in np.arange(n_splits)]
        + [f"Test ROC {l}" for l in np.arange(n_splits)]
        + [f"Test Precision {l}" for l in np.arange(n_splits)]
        + [f"Test Recall {l}" for l in np.arange(n_splits)]
        + [f"Test F1 {l}" for l in np.arange(n_splits)],
    )
    # Go over splits
    for split in range(n_splits):
        print(f"Running split {split+1}/{n_splits}...")
        # First layer of split
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + split)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            # Define training set and test set
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Run grid search on layer 1 training set (second layer of split)
            gsCV = gs_hparam(X_train, y_train, seed=seed)
            performance = record_performance(
                gsCV, X_test, y_test, performance, fold=i, split=split
            )
    # Calculate average across folds
    performance.loc["Average", :] = (
        performance.loc[[f"Fold {l}" for l in np.arange(folds)], :]
        .mean(axis=0)
        .values.ravel()
    )
    performance.to_csv(outfn)


def gs_hparam(X: pd.DataFrame, y: pd.DataFrame, seed: int) -> GridSearchCV:
    """
    Run GridSearch with default model (XGB) and defined hyperparameters.

    Parameters:
        X: input.
        y: target.

    Returns:
        The post-gridsearch GridSearchCV object.
    """
    scaler = StandardScaler()
    clf = xgb.XGBClassifier(
        objective="reg:logistic",
        subsample=1,
        reg_alpha=0,
        reg_lambda=1,
        n_estimators=300,
        seed=seed,
    )
    pipe = Pipeline([("scaler", scaler), ("classifier", clf)])
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
    opt_params = gsCV.best_params_
    opt_mean_score = np.mean(
        gsCV.cv_results_["mean_test_score"][
            np.all(
                [
                    (gsCV.cv_results_[f"param_{param_name}"] == param_vals)
                    for param_name, param_vals in opt_params.items()
                ],
                axis=0,
            )
        ]
    )
    performance.loc[f"Fold {fold}", f"CV ROC {split}"] = opt_mean_score
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
        run(custom_X, custom_y, n_splits=10, outfn=custom_outfn)
