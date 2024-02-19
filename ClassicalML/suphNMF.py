"""
Supervised or unsupervised hybrid non-negative matrix factorization class.
"""

# pylint: disable=no-member
# pylint: disable=too-many-locals
# pylint: disable=attribute-defined-outside-init
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-function-args

from itertools import product
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

loss_type = ["l2", "kl"][0]
eps = 1e-7


class suphNMF(nn.Module):
    """
    Optionally supervised hybrid non-negative matrix factorization class.
    """

    def __init__(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        y: Union[pd.DataFrame, None] = None,
        n_components: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.factorized = False

        # Data
        self.X1 = torch.tensor(X1.values, dtype=torch.float32)
        self.X2 = torch.tensor(X2.values, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32)
            self.supervised = True
        else:
            self.supervised = False

        # Classifier and number of components
        self.n_components = n_components
        self.fc = nn.Linear(in_features=self.n_components, out_features=1)

        # Loss functions
        self.recon_loss_func = nn.MSELoss(reduction="mean")
        pos_weight = self.y.sum() / len(self.y)
        self.clf_loss_func = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=pos_weight
        )

    def forward(self, x):
        """
        Dummy implementation of forwrad.
        This method is not needed for this class since
        the objective is complex and I have implemented my own fit method.
        Since pylint throws a warning for abstract methods that are not
        overridden, so placing this dummy function here.
        """
        return x

    def plus(self):
        """
        Enforce non-negativity on the factorized matrices.
        """
        self.W.data = self.W.data.clamp(min=1e-5)
        self.H1.data = self.H1.data.clamp(min=1e-5)
        self.H2.data = self.H2.data.clamp(min=1e-5)

    def crossval_fit(
        self,
        n_iters: Iterable[int] = [1000, 2000, 3000],
        lrs: Iterable[int] = [1e-4, 1e-3, 1e-2],
        weight_decays: Iterable[float] = [1e-3, 1e-4, 1e-5],
        clf_weights: Iterable[float] = [0, 1e-1, 1e0, 1e1, 1e2],
        ortho_weights: Iterable[float] = [0, 1e-1, 1e0, 1e1, 1e2],
    ):
        """
        Find optimial hyperparameters using cross validation.

        Returns:
            cv_performance: Pandas DataFrame
            best_params: Dict
        """
        ## Hyperparameter ranges
        self.n_iters = n_iters
        self.lrs = lrs
        self.weight_decays = weight_decays
        self.clf_weights = clf_weights
        self.ortho_weights = ortho_weights

        ## Run cross validation
        param_tuples = list(
            product(
                self.n_iters,
                self.lrs,
                self.weight_decays,
                self.clf_weights,
                self.ortho_weights,
            )
        )
        self.cv_performance = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                param_tuples,
                names=("n_iter", "lr", "weight_decay", "clf_weight", "ortho_weight"),
            ),
            columns=["performance"],
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        # Iterate over hparam combinations
        for n_iter, lr, weight_decay, w_clf, w_ortho in param_tuples:
            fold_performances = []
            # Iterate over splits
            for train_index, val_index in skf.split(self.X1, self.y):
                X1_train, X2_train, y_train = (
                    self.X1[train_index, :],
                    self.X2[train_index, :],
                    self.y[train_index],
                )
                X1_val, X2_val, y_val = (
                    self.X1[val_index, :],
                    self.X2[val_index, :],
                    self.y[val_index],
                )
                self.fit(
                    X1_train,
                    X2_train,
                    y_train,
                    n_iter=n_iter,
                    lr=lr,
                    weight_decay=weight_decay,
                    clf_weight=w_clf,
                    ortho_weight=w_ortho,
                )
                eval_metrics = self.evaluate(X1_val, X2_val, y_val)
                # for this given fold and hyperparameters, find the validation performance
                fold_performances.append(eval_metrics["roc_auc"])

            ## Evaluate by calculating mean performance for these hyperparameters
            self.cv_performance.loc[
                (n_iter, lr, weight_decay, w_clf, w_ortho), "performance"
            ] = np.mean(fold_performances)
            self.cv_performance.to_csv(
                "/home/srd6051/test_cv_cleaned_methods.csv"
            )  # TODO: fix hard-coded path # pylint: disable=fixme

        ## Find hyperparameters and n_iter that gave best mean performance across CV folds
        self.cv_performance = self.cv_performance.astype(float)
        best_params_tuple = self.cv_performance.idxmax()
        best_params = dict(zip(self.cv_performance.index.names, best_params_tuple[0]))

        ## Call final fit on whole set with the learned best hyperparameters
        self.fit(self.X1, self.X2, self.y, **best_params)

        self.cv_mean_roc = np.max(self.cv_performance.values)
        self.train_roc = self.evaluate(self.X1, self.X2, self.y)["roc_auc"]

        return self.cv_performance, best_params

    def fit(self, X1, X2, y, n_iter, lr, weight_decay, clf_weight, ortho_weight):
        """
        Fit the NMF.

        Returns:
            None
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Hyperparams
        self.n_iter = n_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.clf_weight = clf_weight
        self.ortho_weight = ortho_weight

        # NMF matrices
        self.W = torch.nn.Parameter(
            torch.normal(
                mean=3.0, std=1.5, size=(X1.shape[0], self.n_components)
            ).clamp(min=1e-5)
        )
        self.H1 = torch.nn.Parameter(
            torch.normal(
                mean=3.0, std=1.5, size=(self.n_components, X1.shape[1])
            ).clamp(min=1e-5)
        )
        self.H2 = torch.nn.Parameter(
            torch.normal(
                mean=3.0, std=1.5, size=(self.n_components, X2.shape[1])
            ).clamp(min=1e-5)
        )

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # training record
        self.clf_roc_record = []
        self.clf_loss_record = []
        self.reconloss1_record = []
        self.reconloss2_record = []
        self.ortholoss_record = []
        self.loss_record = []

        for _ in range(self.n_iter):
            ## Gradient calculation

            # Zero out gradient
            self.optimizer.zero_grad()

            # Calculate factorization (reconstruction) loss (W * H - X)
            loss_recon1 = (
                self.recon_loss_func(torch.mm(self.W, self.H1), X1) / X1.shape[1]
            )  # div by number of features (align scale between X1 and X2)
            loss_recon2 = (
                self.recon_loss_func(torch.mm(self.W, self.H2), X2) / X2.shape[1]
            )

            # Add L2 (or KL) regularization for orthogonality between components
            WtW = torch.mm(torch.t(self.W), self.W)
            if torch.mean(WtW) > 1e-7:  # self.eps
                WtW = WtW / torch.mean(WtW)
            loss_W_ortho = (
                self.recon_loss_func(
                    WtW / self.n_components, torch.eye(self.n_components)
                )
                * self.n_components
            )

            # Add classification loss
            if self.supervised:
                y_out = self.predict(self.W)
                y_score = torch.sigmoid(y_out).ravel()
                loss_clf = self.clf_loss_func(y_out, y)
            else:
                self.loss_clf = torch.tensor(0.0)
            # No need to add L2 regularization to the clf params since added weight decay

            ## Backprop & step
            loss = (
                loss_recon1
                + loss_recon2
                + self.clf_weight * loss_clf
                + self.ortho_weight * loss_W_ortho
            )
            loss.backward()
            self.optimizer.step()
            self.loss_record.append(loss.item())

            ## Set factorized matrix values to positive
            self.plus()

            ## Record performance
            self.reconloss1_record.append(loss_recon1.item())
            self.reconloss2_record.append(loss_recon2.item())
            self.ortholoss_record.append(loss_W_ortho.item())
            if self.supervised:
                self.clf_roc_record.append(
                    metrics.roc_auc_score(y.detach().numpy(), y_score.detach().numpy())
                )
                self.clf_loss_record.append(loss_clf.item())

    def evaluate(self, X1, X2, y):
        """
        Evaluate the model via supervised task.

        Returns:
            eval_metrics: Dict
        """
        eval_metrics = {}

        # Transform input
        W = self.transform(X1, X2)

        # Record losses
        loss_recon1 = self.recon_loss_func(torch.mm(W, self.H1), X1) / X1.shape[1]
        loss_recon2 = self.recon_loss_func(torch.mm(W, self.H2), X2) / X2.shape[1]

        WtW = torch.mm(torch.t(W), W)
        if torch.mean(WtW) > 1e-7:  # self.eps
            WtW = WtW / torch.mean(WtW)
        loss_W_ortho = (
            self.recon_loss_func(WtW / self.n_components, torch.eye(self.n_components))
            * self.n_components
        )

        if self.supervised:
            y_out = self.predict(W)
            y_score = torch.sigmoid(y_out).ravel()
            y_pred = (y_score.clone().detach() > 0.5).int()
            loss_clf = self.clf_loss_func(y_out, y)
            clf_roc = metrics.roc_auc_score(
                y.detach().numpy(), y_score.detach().numpy()
            )
        else:
            y_out = self.predict(W)
            y_score = torch.sigmoid(y_out).ravel()
            y_pred = (y_score.clone().detach() > 0.5).int()
            loss_clf = torch.tensor(0.0)
            clf_roc = metrics.roc_auc_score(
                y.detach().numpy(), y_score.detach().numpy()
            )

        loss_overall = (
            loss_recon1
            + loss_recon2
            + self.clf_weight * loss_clf
            + self.ortho_weight * loss_W_ortho
        )

        eval_metrics["y_pred"] = y_pred
        eval_metrics["y_score"] = y_score
        eval_metrics["roc_auc"] = clf_roc
        eval_metrics["loss_clf"] = loss_clf.item()
        eval_metrics["loss_recon1"] = loss_recon1.item()
        eval_metrics["loss_recon2"] = loss_recon2.item()
        eval_metrics["loss_ortho"] = loss_W_ortho.item()
        eval_metrics["loss_overall"] = loss_overall.item()

        return eval_metrics

    def predict(self, W) -> torch.Tensor:
        """
        Make predictions using the W matrix.

        Returns:
            Predictions
        """
        return self.fc(W)

    def transform(self, X1, X2) -> torch.Tensor:
        """
        Transform a given input X1/X2 into W, using the learned H1/H2.

        Returns:
            The hybrid-factorized W matrix.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        W = torch.nn.Parameter(
            torch.normal(
                mean=3.0, std=1.5, size=(X1.shape[0], self.n_components)
            ).clamp(min=1e-5)
        )
        X1 = torch.tensor(np.array(X1), dtype=torch.float32)
        X2 = torch.tensor(np.array(X2), dtype=torch.float32)
        # if yis not None:
        #    y= torch.tensor(np.array(y), dtype=torch.float32)
        self.optimizer_tf = optim.Adam([W], lr=self.lr, weight_decay=self.weight_decay)
        self.clf_roc_record_tf = []
        self.clf_loss_record_tf = []
        self.reconloss1_record_tf = []
        self.reconloss2_record_tf = []
        self.ortholoss_record_tf = []
        self.loss_record_tf = []

        for _ in range(self.n_iter):
            ## Gradient calculation
            # Zero out gradient
            self.optimizer_tf.zero_grad()
            # Calculate factorization (reconstruction) loss (W * H - X)
            loss_recon1_tf = self.recon_loss_func(torch.mm(W, self.H1), X1)
            loss_recon2_tf = self.recon_loss_func(torch.mm(W, self.H2), X2)
            # Add L2 (or KL) regularization for orthogonality between components
            WtW = torch.mm(torch.t(W), W)
            if torch.mean(WtW) > 1e-7:  # self.eps
                WtW = WtW / torch.mean(WtW)
            loss_W_ortho_tf = (
                self.recon_loss_func(
                    WtW / self.n_components, torch.eye(self.n_components)
                )
                * self.n_components
            )

            ## Backprop & step
            loss = loss_recon1_tf + loss_recon2_tf + self.ortho_weight * loss_W_ortho_tf
            loss.backward()
            self.optimizer_tf.step()
            self.loss_record_tf.append(loss.item())

            ## Set factorized matrix values to positive
            W.data = W.data.clamp(min=1e-5)

            ## Record performance
            # if y is not None:
            #    self.clf_roc_record_tf.append(metrics.roc_auc_score(y, y_pred.detach().numpy()))
            # self.clf_loss_record_tf.append(loss_clf_tf.item())
            self.reconloss1_record_tf.append(loss_recon1_tf.item())
            self.reconloss2_record_tf.append(loss_recon2_tf.item())
            self.ortholoss_record_tf.append(loss_W_ortho_tf.item())

        return W
