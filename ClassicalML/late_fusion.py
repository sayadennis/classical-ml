"""
Wrapper of nested module to accommodate late fusion to the CV process.
"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=no-member

import sys
from typing import Tuple

import nested  # pylint: disable=import-error
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def run_latefusion(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    y: pd.Series,
    outfn: str,
    folds: int = 5,
    seed: int = 38,
    n_splits: int = 5,
) -> None:
    """
    Run the nested cross validation with late fusion to two inputs.

    Parameters:
        - X1: first input
        - X2: second input
        - y: target
        - outfn: CSV file path to record CV results
        - folds: number of folds in CV
        - seed: used in splitting and model initialiaztion
        - n_splits: number of different CV splits to try out

    Returns:
        None (saves performance in CSV)
    """
    X1, X2, y = align_two_inputs(X1, X2, y)
    # Create empty dataframe to record performance
    performance = pd.DataFrame(
        index=[f"Fold {k}" for k in np.arange(folds)] + ["Average"],
        columns=[f"CV ROC {l}" for l in np.arange(n_splits)]
        + [f"Test ROC {l}" for l in np.arange(n_splits)]
        + [f"Test Precision {l}" for l in np.arange(n_splits)]
        + [f"Test Recall {l}" for l in np.arange(n_splits)]
        + [f"Test F1 {l}" for l in np.arange(n_splits)],
    )
    for split in range(n_splits):
        print(f"Running split {split+1}/{n_splits}...")
        # Outer fold
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + split)
        for i, (train_index, test_index) in enumerate(skf.split(X1, y)):
            # Define training set and test set
            X1_train, X1_test = X1.iloc[train_index, :], X1.iloc[test_index, :]
            X2_train, X2_test = X2.iloc[train_index, :], X2.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # Run grid search by an inner fold
            gsCV1 = nested.gs_hparam(X1_train, y_train, seed=seed)
            gsCV2 = nested.gs_hparam(X2_train, y_train, seed=seed)
            performance = record_performance_late_fusion(
                gsCV1, gsCV2, X1_test, X2_test, y_test, performance, fold=i, split=split
            )
    # Calculate average across folds
    performance.loc["Average", :] = (
        performance.loc[[f"Fold {l}" for l in np.arange(folds)], :]
        .mean(axis=0)
        .values.ravel()
    )
    print(f"Writing results to {outfn}...")
    performance.to_csv(outfn)


def align_two_inputs(
    X1: pd.DataFrame, X2: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Align input and target by index values.

    Parameters:
        - Input
        - Target

    Returns:
        - Aligned input
        - Aligned target
    """
    overlap = list(set(X1.index).intersection(set(X2.index)).intersection(set(y.index)))
    X1 = X1.loc[overlap, :]
    X2 = X2.loc[overlap, :]
    y = y.loc[overlap]
    return (X1, X2, y)


def record_performance_late_fusion(
    gsCV1: GridSearchCV,
    gsCV2: GridSearchCV,
    X_test1: pd.DataFrame,
    X_test2: pd.DataFrame,
    y_test: pd.DataFrame,
    performance: pd.DataFrame,
    fold: int,
    split: int,
) -> pd.DataFrame:
    """
    Record performance of the grid search process.
    """
    # First record CV performance
    _, opt_mean_score1 = nested.find_optimal(gsCV1)
    _, opt_mean_score2 = nested.find_optimal(gsCV2)
    performance.loc[f"Fold {fold}", f"CV ROC {split}"] = np.mean(
        [opt_mean_score1, opt_mean_score2]
    )
    # Next record test set performance
    y_prob1 = gsCV1.predict_proba(X_test1)
    y_prob2 = gsCV2.predict_proba(X_test2)
    if y_prob1.shape[1] == 2:  # if binary class
        y_prob1 = y_prob1[:, 1]
        y_prob2 = y_prob2[:, 1]
    y_prob = np.concatenate(
        (y_prob1.reshape(-1, 1), y_prob2.reshape(-1, 1)), axis=1
    ).mean(axis=1)
    y_pred = y_prob.round()
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
        run_latefusion(
            iris_X, iris_X, iris_y, n_splits=2, outfn="test_iris_late_fusion.csv"
        )
    else:
        custom_X1 = pd.read_csv(args[1], index_col=0, header=0)
        custom_X2 = pd.read_csv(args[2], index_col=0, header=0)
        custom_y = pd.read_csv(args[3], index_col=0, header=0)
        custom_outfn = args[4]
        N_CPU = int(args[5])
        run_latefusion(custom_X1, custom_X2, custom_y, n_splits=5, outfn=custom_outfn)
