# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

"""
Module: ClassicalML

This module provides the class to perform cross-validation across multiple models.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.decomposition import NMF
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# from numpy import interp

with open("param_distributions.json", "r", encoding="utf-8") as f:
    PARAM_DIST = json.load(f)


class ClassicalML:
    """
    ClassicalML

    This class helps perform cross-validation across multiple models.
    """

    def __init__(
        self,
        scoring_metric="accuracy",
        nmf: bool = False,
        standardscale: bool = True,
        n_cpu: int = 8,
        seed: int = 42,
    ):
        """
        Initialize the ClassicalML object with the given parameters.

        Parameters:
        - scoring_metric: scoring metric to use. Select from sklearn.metrics.get_scorer_names()
        - nmf: indicate whether to include non-negative matrix factorization.
        - standardscale: indicate whether to scale data
        - n_cpu: number of CPUs to use for training
        - seed: used for splits, model initializations etc.
        """
        super(ClassicalML).__init__()
        self.n_cpu = n_cpu
        self.seed = seed
        self.scoring_metric = scoring_metric
        self.nmf = nmf
        self.standardscale = standardscale
        if self.nmf & self.standardscale:
            raise ValueError(
                "Cannot have both nmf=True and standardscale=True because "
                "standard-scaling results in negative values."
            )
        self.models = {
            "LRM": LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=3000,
                random_state=self.seed,
            ),
            "LASSO": LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=3000,
                random_state=self.seed,
            ),
            "ElasticNet": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=3000,
                random_state=self.seed,
            ),
            "SVM": SVC(
                class_weight="balanced",
                max_iter=3000,
                probability=True,
                random_state=self.seed,
            ),  # , decision_function_shape='ovr'
            "RF": RandomForestClassifier(
                n_estimators=300,
                criterion="gini",
                class_weight="balanced",
                random_state=self.seed,
            ),
            "GB": GradientBoostingClassifier(subsample=0.8, random_state=self.seed),
            "XGB": xgb.XGBClassifier(
                objective="reg:logistic",
                subsample=1,
                reg_alpha=0,
                reg_lambda=1,
                n_estimators=300,
                seed=self.seed,
            ),
        }
        self.best_model_name = None
        self.best_model = None

    def _confirm_numpy(self, X, y):
        """
        Function that will convert inputs into numpy if needed.

        Parameters:
        - X: input matrix in pandas DataFrame or numpy array.
        - y: target array in pamdas DataFrame or numpy array.

        Returns:
        - X: input matrix in numpy array
        - y: target array as numpy 1D array
        """
        # Function that will convert inputs into compatible format
        if isinstance(X, np.ndarray):
            X = X.to_numpy()
        else:
            X = np.array(X)
        if isinstance(y, np.ndarray):
            y = y.to_numpy(dtype=int)
        else:
            y = np.array(y, dtype=int)
        y = y.ravel()  # convert to 1D array
        return X, y

    def record_tuning(self, X_train, y_train, X_test, y_test, outfn, multiclass=False):
        """
        Function that will take a given train and test set to
        calculate CV scores and attach best model as a class attribute.

        Parameters:
        - X_train:
        - y_train:
        - X_test:
        - y_test:
        - outfn:
        - multiclass:

        Returns: None
        """
        self.feature_names = list(X_train.columns)
        if self.nmf:
            max_k = (X_train.shape[0] * X_train.shape[1]) // (
                X_train.shape[0] + X_train.shape[1]
            )
            # If max_k > max(X.shape[0]), lower max_k
            max_k = np.min((max_k, X_train.shape[0], X_train.shape[1]))
            if max_k >= 500:
                self.nmf_params = {
                    "nmf__n_components": list(np.arange(50, 501, step=50))
                }
            elif max_k < 25:
                self.nmf_params = {
                    "nmf__n_components": list(np.arange(2, max_k + 1, 2))
                }
            else:
                max_k = (
                    max_k // 100
                ) * 100  # for example, if max_k=467, this sets max_k=400
                self.nmf_params = {
                    "nmf__n_components": list(np.arange(25, max_k + 1, 25))
                }
        else:
            self.nmf_params = {}
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()
        cv_record = pd.DataFrame(
            None,
            index=self.models.keys(),
            columns=[
                "opt_params",
                f"crossval_{self.scoring_metric}",
                "test_bal_acc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
            ],
        )
        trained_models = {}
        for model_name, _ in self.models.items():
            opt_params, opt_score, clf = self.clf_crossval(
                X_train, y_train, model_name=model_name
            )
            trained_models[model_name] = clf
            cv_record.loc[model_name]["opt_params"] = str(opt_params)
            cv_record.loc[model_name][f"crossval_{self.scoring_metric}"] = opt_score
            pf_dict = self.evaluate_model(clf, X_test, y_test, multiclass=multiclass)
            for pf_key, pf_val in pf_dict.items():
                cv_record.loc[model_name][pf_key] = pf_val
        cv_record.to_csv(outfn, header=True, index=True)
        # save best performing model
        self.best_model_name = cv_record.iloc[
            np.argmax(cv_record[f"crossval_{self.scoring_metric}"])
        ].name
        self.best_model = trained_models[self.best_model_name]

    def clf_crossval(self, X_train, y_train, model_name):
        """
        Perform cross-validation for a given input, target, and model.

        Parameters:
        - X_train:
        - y_train:
        - model_name:

        Returns:
        - Optimal parameter combinations
        - Best mean cross-validation score with the above parameters
        - The best classifier.
        """
        if self.standardscale:
            scaler = StandardScaler()
            clf = self.models[model_name]
            pipe = Pipeline([("scaler", scaler), ("classifier", clf)])
        elif self.nmf:
            nmf = NMF(init="nndsvd", random_state=24)
            clf = self.models[model_name]
            pipe = Pipeline([("nmf", nmf), ("classifier", clf)])
        else:
            clf = self.models[model_name]
            pipe = Pipeline([("classifier", clf)])
        #
        gsCV = GridSearchCV(
            pipe,
            param_grid=(self.nmf_params | PARAM_DIST[model_name]),
            n_jobs=self.n_cpu,
            scoring=self.scoring_metric,
            refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True),
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
        return opt_params, opt_mean_score, gsCV.best_estimator_

    ## Function that will take classifier and evaluate on test set
    def evaluate_model(self, clf, X_test, y_test, multiclass=False):
        """
        Evaluate the trained model on the validation or test set.

        Parameters:
        - clf: trained estimator
        - X_test: input of the test/validation set
        - y_test: target of the test/validation set

        Returns:
        - Performance dictionary with evaluation metrics
        """
        pf_dict = {}  # performance dictionary
        if not multiclass:
            y_pred = np.array(
                clf.predict(X_test).round(), dtype=int
            )  # round b/c Lasso/ElasticNet outputs continuous values with .predict()
            y_prob = clf.predict_proba(X_test)[:, 1]
            pf_dict["test_bal_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)
            pf_dict["roc_auc"] = metrics.roc_auc_score(y_test, y_prob)
            pf_dict["precision"] = metrics.precision_score(y_test, y_pred)
            pf_dict["recall"] = metrics.recall_score(y_test, y_pred)
            pf_dict["f1"] = metrics.f1_score(y_test, y_pred)
        else:
            y_pred = np.array(
                clf.predict(X_test).round(), dtype=int
            )  # 1D array (LR, Lasso, SVM)
            y_prob = clf.predict_proba(X_test)  # gives a n by n_class matrix
            y_test_onehot = (
                OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
            )
            ## Micro-averaged ROC
            # fpr, tpr, _ = metrics.roc_curve(y_test_onehot.ravel(), y_prob.ravel())
            # roc_auc = metrics.auc(fpr, tpr)
            ## Macro-averaged ROC ??
            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            # for i in range(3): # 3 = n_classes
            #     fpr[i], tpr[i], _ = metrics.roc_curve(y_test_onehot[:,i], y_prob[:,i])
            #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)])) # 3 = n_classes
            # mean_tpr = np.zeros_like(all_fpr)
            # for i in range(3):
            #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # mean_tpr /= 3
            # roc_auc = metrics.auc(all_fpr, mean_tpr)
            # Record metrics in performance dictionary
            pf_dict["test_bal_acc"] = metrics.balanced_accuracy_score(y_test, y_pred)
            pf_dict["roc_auc"] = metrics.roc_auc_score(y_test_onehot, y_prob)  # roc_auc
            pf_dict["precision"] = metrics.precision_score(
                y_test, y_pred, average="macro"
            )
            pf_dict["recall"] = metrics.recall_score(y_test, y_pred, average="macro")
            pf_dict["f1"] = metrics.f1_score(y_test, y_pred, average="macro")
        return pf_dict
