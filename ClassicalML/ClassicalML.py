import os
import sys
import numpy as np
import pandas as pd

# classifiers 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
# from sklearn.neural_network import MLPClassifier

# tools for cross-validation 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from numpy import interp

# dimensionality reduction 
from sklearn.decomposition import NMF

# visualization
import matplotlib.pyplot as plt

class ClassicalML():
    def __init__(self, scoring_metric='accuracy', nmf=False, standardscale=True, seed=42):
        super(ClassicalML).__init__()
        self.seed = seed
        self.scoring_metric = scoring_metric
        self.nmf = nmf
        self.standardscale = standardscale
        if (self.nmf & self.standardscale):
            raise ValueError('Cannot have both nmf=True and standardscale=True because ' \
                             'standard-scaling results in negative values.')
        self.models = {
            'LRM' : LogisticRegression(
                penalty='l2', class_weight='balanced', max_iter=3000, random_state=self.seed
            ),
            'LASSO' : LogisticRegression(
                penalty='l1', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=self.seed
            ),
            'ElasticNet' : LogisticRegression(
                penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=3000, random_state=self.seed
            ),
            'SVM' : SVC(
                class_weight='balanced', max_iter=3000, probability=True, random_state=self.seed
            ), #, decision_function_shape='ovr'
            'RF' : RandomForestClassifier(
                n_estimators=300, criterion='gini', max_features='auto', 
                class_weight='balanced', n_jobs=8, random_state=self.seed
            ),
            'GB' : GradientBoostingClassifier(
                subsample=0.8, random_state=self.seed
            ),
            'XGB' : xgb.XGBClassifier(
                objective='reg:logistic', subsample=1, reg_alpha=0, reg_lambda=1, n_estimators=300, seed=self.seed
            ),
        }
        self.model_params = {
            'LRM' : {
                'classifier__C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
            },
            'LASSO' : {
                'classifier__C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
            },
            'ElasticNet' : {
                'classifier__C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3],
                'classifier__l1_ratio' : np.arange(0.1, 1., step=0.1)
            },
            'SVM' : {
                'classifier__kernel' : ['linear', 'rbf'], 
                'classifier__C' : [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3],
                'classifier__gamma' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
            }, #, decision_function_shape='ovr'
            'RF' : {
                # 'criterion' : ['gini', 'entropy'],
                'classifier__max_depth' : [3, 5, 10, 25, 50], # or could set min_samples_split 
                'classifier__min_samples_leaf' : [2, 4, 6, 8, 10, 15, 20]
            },
            'GB' : {
                'classifier__loss' : ['deviance', 'exponential'],
                'classifier__min_samples_split' : [2, 6, 10, 15, 20],
                'classifier__max_depth' : [5, 10, 25, 50, 75]
            },
            'XGB' : {
                'classifier__learning_rate' : [0.01, 0.1, 0.2],
                'classifier__max_depth' : [5, 10, 15, 25],
                'classifier__colsample_bytree' : [0.3, 0.5, 0.7, 0.9],
                'classifier__gamma' : [0.1, 1, 2]
            },
        }
        self.best_model_name = None
        self.best_model = None
    
    def _confirm_numpy(self, x):
        # Function that will convert inputs into compatible format 
        if type(x) != np.ndarray:
            x = x.to_numpy()
        else:
            x = np.array(x)
        if type(y) != np.ndarray:
            y = y.to_numpy(dtype=int)
        else:
            y = np.array(y, dtype=int)
        y = y.ravel() # convert to 1D array 
        return X, y

    ## Function that will take X_train, y_train, run all the hyperparameter tuning, and record cross-validation performance 
    def record_tuning(self, X_train, y_train, X_test, y_test, outfn, multiclass=False):
        self.feature_names = list(X_train.columns)
        if self.nmf:
            max_k = (X_train.shape[0] * X_train.shape[1]) // (X_train.shape[0] + X_train.shape[1])
            # If max_k is larger than the number of rows or columns of X, lower max_k to that minimum value
            max_k = np.min((max_k, X_train.shape[0], X_train.shape[1]))
            if max_k >= 500:
                self.nmf_params = {'nmf__n_components' : [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]}
            else:
                max_k = (max_k // 100) * 100 # for example, if max_k=467, this sets max_k=400
                self.nmf_params = {'nmf__n_components' : list(np.arange(25, max_k+1, 25))}
        else:
            self.nmf_params = {}
        y_train = np.array(y_train).ravel(); y_test = np.array(y_test).ravel()
        cv_record = pd.DataFrame(
            None, index=self.models.keys(), 
            columns=['opt_params', f'crossval_{self.scoring_metric}', 'test_bal_acc', 'roc_auc', 'precision', 'recall', 'f1']
        )
        trained_models = {}
        for model_name in self.models.keys():
            opt_params, opt_score, clf = self.clf_crossval(X_train, y_train, model_name=model_name)
            trained_models[model_name] = clf
            cv_record.loc[model_name]['opt_params'] = str(opt_params)
            cv_record.loc[model_name][f'crossval_{self.scoring_metric}'] = opt_score
            pf_dict = self.evaluate_model(clf, X_test, y_test, multiclass=multiclass)
            for pf_key in pf_dict:
                cv_record.loc[model_name][pf_key] = pf_dict[pf_key]
        cv_record.to_csv(outfn, header=True, index=True)
        # save best performing model
        self.best_model_name = cv_record.iloc[np.argmax(cv_record[f'crossval_{self.scoring_metric}'])].name
        self.best_model = trained_models[self.best_model_name]

    def clf_crossval(self, X_train, y_train, model_name):
        if self.standardscale:
            scaler = StandardScaler()
            clf = self.models[model_name]
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = self.models[model_name]
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
        else:
            clf = self.models[model_name]
            pipe = Pipeline([('classifier', clf)])
        # 
        gsCV = GridSearchCV(
            pipe,
            param_grid=(self.nmf_params | self.model_params[model_name]), n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)
        )
        gsCV.fit(X_train, y_train)
        opt_params = gsCV.best_params_
        opt_mean_score = np.mean(gsCV.cv_results_['mean_test_score'][
            np.all([(gsCV.cv_results_[f'param_{param_name}'] == opt_params[param_name]) for param_name in opt_params.keys()], axis=0)
        ])
        return opt_params, opt_mean_score, gsCV.best_estimator_

    ## Function that will take classifier and evaluate on test set
    def evaluate_model(self, clf, X_test, y_test, multiclass=False):
        pf_dict = {} # performance dictionary
        if multiclass == False:
            y_pred = np.array(clf.predict(X_test).round(), dtype=int) # round with dtype=int b/c Lasso and ElasticNet outputs continuous values with the .predict() method
            y_prob = clf.predict_proba(X_test)[:,1]
            pf_dict['test_bal_acc'] = metrics.balanced_accuracy_score(y_test, y_pred)
            pf_dict['roc_auc'] = metrics.roc_auc_score(y_test, y_prob)
            pf_dict['precision'] = metrics.precision_score(y_test, y_pred)
            pf_dict['recall'] = metrics.recall_score(y_test, y_pred)
            pf_dict['f1'] = metrics.f1_score(y_test, y_pred)
        else:
            y_pred = np.array(clf.predict(X_test).round(), dtype=int) # 1D array (LR, Lasso, SVM)
            y_prob = clf.predict_proba(X_test) # gives a n by n_class matrix 
            y_test_onehot = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
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
            pf_dict['test_bal_acc'] = metrics.balanced_accuracy_score(y_test, y_pred)
            pf_dict['roc_auc'] = metrics.roc_auc_score(y_test_onehot, y_prob) # roc_auc 
            pf_dict['precision'] = metrics.precision_score(y_test, y_pred, average='macro')
            pf_dict['recall'] = metrics.recall_score(y_test, y_pred, average='macro')
            pf_dict['f1'] = metrics.f1_score(y_test, y_pred, average='macro')
        return pf_dict
