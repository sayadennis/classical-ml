"""
Wrapper of nested module to incorporate optionally supervised hybrid NMF to the nested CV process.
"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=no-member

import sys
from typing import Iterable, Tuple

import nested  # pylint: disable=import-error
import numpy as np
import pandas as pd
import torch
from late_fusion import align_two_inputs  # pylint: disable=import-error
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from suphNMF import suphNMF  # pylint: disable=import-error


def run_nmf(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    y: pd.Series,
    outfn: str,
    folds: int = 5,
    seed: int = 38,
    n_splits: int = 5,
) -> None:
    """
    Run the nested cross validation with optionally supervised hybrid NMF.

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
            # Scale all data based on the train set
            scaler = MinMaxScaler(feature_range=(0, 30))
            X1_train = pd.DataFrame(
                scaler.fit_transform(X1_train),
                index=X1_train.index,
                columns=X1_train.columns,
            )
            X1_test = pd.DataFrame(
                scaler.transform(X1_test), index=X1_test.index, columns=X1_test.columns
            )
            X2_train = pd.DataFrame(
                scaler.fit_transform(X2_train),
                index=X2_train.index,
                columns=X2_train.columns,
            )
            X2_test = pd.DataFrame(
                scaler.transform(X2_test), index=X2_test.index, columns=X2_test.columns
            )
            # Find best hyper parameters for NMF
            _, W_train, nmf_obj = crossval_hnmf(X1_train, X2_train, y_train)
            # Run grid search by an inner fold
            W_train = pd.DataFrame(W_train.detach().numpy(), index=X1_train.index)
            gsCV = nested.gs_hparam(
                W_train, y_train, seed=seed, scale=False
            )  # already standard scaled
            W_test = nmf_obj.transform(X1_test, X2_test)
            W_test = pd.DataFrame(W_test.detach().numpy(), index=X1_test.index)
            performance = nested.record_performance(
                gsCV, W_test, y_test, performance, fold=i, split=split
            )
    # Calculate average across folds
    performance.loc["Average", :] = (
        performance.loc[[f"Fold {l}" for l in np.arange(folds)], :]
        .mean(axis=0)
        .values.ravel()
    )
    print(f"Writing results to {outfn}...")
    performance.to_csv(outfn)


def crossval_hnmf(
    X1_train: pd.DataFrame,
    X2_train: pd.DataFrame,
    y_train: pd.Series,
    ks: Iterable[int] = np.arange(4, 12),
) -> Tuple[dict, pd.DataFrame, suphNMF,]:
    """
    Run cross-validation with hybrid NMF.

    Parameters:
        - Input 1
        - Input 2
        - Target
        - Number of NMF components to iterate over

    Returns:
        - Dictionary of best hNMF parameters
        - Factorized W of the used training set
        - hNMF object
    """
    nmf_record = {}
    for k in ks:
        print(f"Trying k={k}...")
        nmf = suphNMF(X1_train, X2_train, y_train, n_components=k)
        cv_results, best_params = nmf.crossval_fit(
            n_iters=[1000, 2000],
            lrs=[1e-2],
            clf_weights=[0.0],
            ortho_weights=[0.0, 1e-1, 1e0, 1e1],
        )
        nmf_record[k] = {
            "best_params": best_params,
            "cv_mean_roc": nmf.cv_mean_roc,
            "cv_results": cv_results,
        }
    best_k = np.arange(4, 12)[
        np.argmax([nmf_record[k]["cv_mean_roc"] for k in range(4, 12)])
    ]
    print(f"Finished testing all Ks! Best k was {best_k} for this round.")
    best_params = nmf_record[best_k]["best_params"]
    # Refit with learned parameters
    nmf = suphNMF(X1_train, X2_train, y_train, n_components=best_k)
    nmf.fit(
        torch.tensor(X1_train.values, dtype=torch.float32),
        torch.tensor(X2_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        **best_params,
    )
    W_train = nmf.transform(X1_train, X2_train)
    print(f"Finished final fit with best k={best_k}!")
    return (best_params, W_train, nmf)


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:  # no arguments
        iris_X, iris_y = load_iris(return_X_y=True, as_frame=True)
        run_nmf(iris_X, iris_X, iris_y, n_splits=2, outfn="test_iris_nmf.csv")
    else:
        custom_X1 = pd.read_csv(args[1], index_col=0, header=0)
        custom_X2 = pd.read_csv(args[2], index_col=0, header=0)
        custom_y = pd.read_csv(args[3], index_col=0, header=0)
        custom_outfn = args[4]
        N_CPU = int(args[5])
        run_nmf(custom_X1, custom_X2, custom_y, n_splits=5, outfn=custom_outfn)
