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
        index=[f"fold {k}" for k in np.arange(folds)] + ["average"],
        columns=[f"CV {l}" for l in np.arange(n_splits)]
        + [f"test {l}" for l in np.arange(n_splits)],
    )
    # Go over splits
    for split in range(n_splits):
        print(f"Running split {split+1}/{n_splits}...")
        # First layer of split
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + split)
        cv_scores = []
        test_scores = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            # Define training set and test set
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Run grid search on layer 1 training set (second layer of split)
            scaler = StandardScaler()
            # clf = RandomForestClassifier(
            #    n_estimators=300,
            #    criterion="gini",
            #    class_weight="balanced",
            #    random_state=seed,
            # )
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
            gsCV.fit(X_train, y_train)
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
            performance.loc[f"fold {i}", f"CV {split}"] = opt_mean_score
            cv_scores.append(opt_mean_score)
            y_pred = gsCV.predict_proba(X_test)
            if y_pred.shape[1] == 2:  # if binary class
                y_pred = y_pred[:, 1]
            test_score = metrics.roc_auc_score(y_test, y_pred, multi_class="ovr")
            performance.loc[f"fold {i}", f"test {split}"] = test_score
            test_scores.append(test_score)
        # Average test scores across the initial 5 folds
        avg_cv_score = np.mean(cv_scores)
        performance.loc["average", f"CV {split}"] = avg_cv_score
        avg_test_score = np.mean(test_scores)
        performance.loc["average", f"test {split}"] = avg_test_score
    performance.loc["average", :] = (
        performance.loc[[f"fold {l}" for l in np.arange(folds)], :]
        .mean(axis=0)
        .values.ravel()
    )
    performance.to_csv(outfn)


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
