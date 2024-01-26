# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

"""
Module: nested

This module provides tools for nested cross validation.
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_CPU = 4
SCORING_METRIC = "roc_auc_ovr"
PARAM_DIST = {
    "classifier__max_depth": [3, 5, 10, 25, 50],
    "classifier__min_samples_leaf": [2, 4, 6, 8, 10, 15, 20],
}


def run(
    X: pd.DataFrame,
    y: pd.DataFrame,
    folds: int = 5,
    seed: int = 8,  # n_splits: int = 1
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
    # First layer of split
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cv_scores = []
    test_scores = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Define training set and test set
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Run grid search on layer 1 training set (second layer of split)
        scaler = StandardScaler()
        clf = RandomForestClassifier()
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
        print(f"Optimal mean score for fold {i}: {opt_mean_score}")
        cv_scores.append(opt_mean_score)
        y_pred = gsCV.predict_proba(X_test)
        test_scores.append(metrics.roc_auc_score(y_test, y_pred, multi_class="ovr"))
    # Average test scores across the initial 5 folds
    avg_cv_score = np.mean(cv_scores)
    print(f"Average CV score: {avg_cv_score}")
    avg_test_score = np.mean(test_scores)
    print(f"Average test score: {avg_test_score}")


if __name__ == "__main__":
    iris_X, iris_y = load_iris(return_X_y=True, as_frame=True)
    run(iris_X, iris_y)
