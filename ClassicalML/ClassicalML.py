import os
import sys
import numpy as np
import pandas as pd

# classifiers 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
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
    def __init__(self, scoring_metric='accuracy', nmf=False):
        super(ClassicalML).__init__()
        self.scoring_metric = scoring_metric
        self.nmf = nmf
        self.lrm_params = {
            'C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
        }
        self.lasso_params = {
            'C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
        }
        self.elasticnet_params = {
            'C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3],
            'l1_ratio' : np.arange(0.1, 1., step=0.1)
        }
        self.svm_params = {
            'kernel' : ['linear', 'rbf'], 
            'C' : [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3],
            'gamma' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
        }
        self.rf_params = {
            # 'criterion' : ['gini', 'entropy'],
            'max_depth' : [3, 5, 10, 25, 50], # or could set min_samples_split 
            'min_samples_leaf' : [2, 4, 6, 8, 10, 15, 20]
        }
        self.gb_params = {
            'loss' : ['deviance', 'exponential'],
            'min_samples_split' : [2, 6, 10, 15, 20],
            'max_depth' : [5, 10, 25, 50, 75]
        }
        self.xgb_params = {
            'learning_rate' : [0.01, 0.1, 0.2],
            'max_depth' : [5, 10, 15, 25],
            'colsample_bytree' : [0.3, 0.5, 0.7, 0.9],
            'gamma' : [0.1, 1, 2]
        }
        self.best_model_name = None
        self.best_model = None
    
    def confirm_numpy(self, X, y):
        # Function that will convert inputs into compatible format 
        if type(X) != np.ndarray:
            X = X.to_numpy(dtype=int)
        else:
            X = np.array(X, dtype=int)
        if type(y) != np.ndarray:
            y = y.to_numpy(dtype=int)
        else:
            y = np.array(y, dtype=int)
        y = y.ravel() # convert to 1D array 
        return X, y

    # Logistic regression 
    def lrm_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        gsLR = GridSearchCV(
            LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000, random_state=seed),
            param_grid=self.lrm_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsLR.fit(X_train, y_train)
        opt_params = gsLR.best_params_
        opt_mean_score = np.mean(
            gsLR.cv_results_['mean_test_score'][
                (gsLR.cv_results_['param_C'] == opt_params['C'])
            ]
        )
        return opt_params, opt_mean_score, gsLR

    # LASSO cross-validation
    def lasso_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        gsLS = GridSearchCV(
            LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=1000, random_state=seed),
            param_grid=self.lasso_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsLS.fit(X_train, y_train)
        opt_params = gsLS.best_params_
        opt_mean_score = np.mean(
            gsLS.cv_results_['mean_test_score'][
                (gsLS.cv_results_['param_C'] == opt_params['C'])
            ]
        )
        return opt_params, opt_mean_score, gsLS

    # Elastic Net cross-validation 
    def elasticnet_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        gsEN = GridSearchCV(
            LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=1000, random_state=seed),
            param_grid=self.elasticnet_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsEN.fit(X_train, y_train)
        opt_params = gsEN.best_params_
        opt_mean_score = np.mean(
            gsEN.cv_results_['mean_test_score'][
                (gsEN.cv_results_['param_C'] == opt_params['C']) &
                (gsEN.cv_results_['param_l1_ratio'] == opt_params['l1_ratio'])
            ]
        )
        return opt_params, opt_mean_score, gsEN

    # Support vector machine cross-validation 
    def svm_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        gsCV = GridSearchCV(
            SVC(class_weight='balanced', max_iter=1000, probability=True, random_state=seed),  #, decision_function_shape='ovr'
            param_grid=self.svm_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsCV.fit(X_train, y_train)
        opt_params = gsCV.best_params_
        opt_mean_score = np.mean(
            gsCV.cv_results_['mean_test_score'][
                (gsCV.cv_results_['param_kernel'] == opt_params['kernel']) & 
                (gsCV.cv_results_['param_C'] == opt_params['C']) & 
                (gsCV.cv_results_['param_gamma'] == opt_params['gamma'])
            ]
        )
        return opt_params, opt_mean_score, gsCV

    # Random Forest cross-validation
    def rf_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        gsRF = GridSearchCV(
            RandomForestClassifier(n_estimators=300, criterion='gini', max_features='auto', class_weight='balanced', n_jobs=8, random_state=seed),
            param_grid=self.rf_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsRF.fit(X_train, y_train)
        opt_params = gsRF.best_params_
        opt_mean_score = np.mean(
            gsRF.cv_results_['mean_test_score'][
                (gsRF.cv_results_['param_max_depth'] == opt_params['max_depth']) &
                (gsRF.cv_results_['param_min_samples_leaf'] == opt_params['min_samples_leaf'])
            ]
        )
        return opt_params, opt_mean_score, gsRF

    # Gradient Boosting cross-validation
    def gb_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        gsGB = GridSearchCV(
            GradientBoostingClassifier(subsample=0.8, random_state=seed),
            param_grid=self.gb_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsGB.fit(X_train, y_train)
        opt_params = gsGB.best_params_
        opt_mean_score = np.mean(
            gsGB.cv_results_['mean_test_score'][
                (gsGB.cv_results_['param_loss'] == opt_params['loss']) &
                (gsGB.cv_results_['param_min_samples_split'] == opt_params['min_samples_split']) &
                (gsGB.cv_results_['param_max_depth'] == opt_params['max_depth'])
            ]
        )
        return opt_params, opt_mean_score, gsGB

    # XGBoost
    def xgb_cv(self, X_train, y_train, seed=0):
        # X_train, y_train = self.confirm_numpy(X_train, y_train)
        # use regular GridSearchCV
        gsXGB = GridSearchCV(
            xgb.XGBClassifier(objective='reg:logistic', subsample=1, reg_alpha=0, reg_lambda=1, n_estimators=300, seed=seed),
            param_grid=self.xgb_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
            cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        )
        gsXGB.fit(X_train, y_train)
        opt_params = gsXGB.best_params_
        opt_mean_score = np.mean(
            gsXGB.cv_results_['mean_test_score'][
                (gsXGB.cv_results_['param_learning_rate'] == opt_params['learning_rate']) &
                (gsXGB.cv_results_['param_max_depth'] == opt_params['max_depth']) &
                (gsXGB.cv_results_['param_colsample_bytree'] == opt_params['colsample_bytree']) &
                (gsXGB.cv_results_['param_gamma'] == opt_params['gamma'])
            ]
        )
        xgb_clf = xgb.XGBClassifier(
            objective='reg:logistic', subsample=1, reg_alpha=0, reg_lambda=1, n_estimators=300, seed=seed,
            learning_rate=opt_params['learning_rate'], max_depth=opt_params['max_depth'],
            colsample_bytree=opt_params['colsample_bytree'], gamma=opt_params['gamma']
        )
        xgb_clf.fit(X_train, y_train)
        return opt_params, opt_mean_score, xgb_clf

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

    ## Function that will take X_train, y_train, run all the hyperparameter tuning, and record cross-validation performance 
    def record_tuning(self, X_train, y_train, X_test, y_test, outfn, multiclass=False):
        X_train, y_train = self.confirm_numpy(X_train, y_train)
        X_test, y_test = self.confirm_numpy(X_test, y_test)
        tuner_dict = {
            'LRM' : self.lrm_cv,
            'SVM' : self.svm_cv,
            'LASSO' : self.lasso_cv,
            'ElasticNet' : self.elasticnet_cv,
            'RF' : self.rf_cv,
            'GB' : self.gb_cv,
            'XGB' : self.xgb_cv
        }
        cv_record = pd.DataFrame(
            None, index=tuner_dict.keys(), 
            columns=['opt_params', f'crossval_{self.scoring_metric}', 'test_bal_acc', 'roc_auc', 'precision', 'recall', 'f1']
        )
        trained_models = {}
        for model_key in tuner_dict.keys():
            opt_params, opt_score, clf = tuner_dict[model_key](X_train, y_train)
            trained_models[model_key] = clf
            cv_record.loc[model_key]['opt_params'] = str(opt_params)
            cv_record.loc[model_key][f'crossval_{self.scoring_metric}'] = opt_score
            pf_dict = self.evaluate_model(clf, X_test, y_test, multiclass=multiclass)
            for pf_key in pf_dict:
                cv_record.loc[model_key][pf_key] = pf_dict[pf_key]
        cv_record.to_csv(outfn, header=True, index=True)
        # save best performing model
        self.best_model_name = cv_record.iloc[np.argmax(cv_record[f'crossval_{self.scoring_metric}'])].name
        self.best_model = trained_models[self.best_model_name]
        return

    ## Below is for NMF ## 
    def get_F(self, k, X_train, y_train, X_test, y_test):
        train_size = X_train.shape[0] # rows are patients for X
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        A = X.T
        nmf = NMF(n_components=k, init='nndsvd', random_state=24)
        W = nmf.fit_transform(A)
        F = np.dot(A.T, W)
        F_train = F[:train_size, :]
        F_test = F[train_size:, :]

        return F_train, F_test, W

    def coef_genes(self, W, genes, thres=0.01): # length of genes and rows of W should match
        if W.shape[0] != len(genes):
            return -1
        else:
            group_list = []
            for i in range(W.shape[1]): # iterate through columns of W i.e. weight vectors of each factor groups 
                coef_genes = genes[np.where(W[:,i] > thres)[0]]
                genes_final = []
                for j in range(len(coef_genes)):
                    split_list = coef_genes[j].split(';') # element in list can contain multiple gene names (overlapping)
                    for gene in split_list:
                        if gene in genes_final:
                            continue
                        else:
                            genes_final.append(gene)
                group_list.append([genes_final])
        return genes_final

    def record_tuning_NMF(self, X_train, y_train, X_test, y_test, outfn, k_list=None, multiclass=False):
        # define k_list if not already explicitly defined
        if k_list is None:
            max_k = (X_train.shape[0] * X_train.shape[1]) // (X_train.shape[0] * X_train.shape[1])
            # If max_k is larger than the number of rows or columns of X, lower max_k to that minimum value
            max_k = np.min((max_k, X_train.shape[0], X_train.shape[1]))
            if max_k >= 500:
                k_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
            else:
                max_k = (max_k // 100) * 100 # for example, if max_k=467, this sets max_k=400
                k_list = list(np.arange(25, max_k+1, 25))
        
        tuner_dict = {
            'LRM' : self.lrm_cv,
            'SVM' : self.svm_cv,
            'LASSO' : self.lasso_cv,
            'ElasticNet' : self.elasticnet_cv,
            'RF' : self.rf_cv,
            'GB' : self.gb_cv,
            'XGB' : self.xgb_cv
        }

        cv_record = pd.DataFrame(
            None, index=np.arange(len(list(tuner_dict))*len(k_list)),
            columns=['model', 'NMF_k', 'opt_params', f'crossval_{self.scoring_metric}', 'test_bal_acc', 'roc_auc', 'precision', 'recall', 'f1']
        )

        # fill in models and NMF_k by all possible combinations
        cv_record['model'] = np.repeat(list(tuner_dict.keys()), repeats=len(k_list))
        cv_record['NMF_k'] = np.repeat(np.array(k_list).reshape((-1,1)), repeats=len(list(tuner_dict.keys())), axis=1).T.ravel()

        for k in k_list:
            F_train, F_test, _ = self.get_F(k, X_train, y_train, X_test, y_test)
            for model_key in tuner_dict.keys():
                opt_params, opt_score, clf = tuner_dict[model_key](F_train, y_train)
                bool_loc = np.array(
                    [x == model_key for x in cv_record['model']]
                ) & np.array(
                    [x == k for x in cv_record['NMF_k']]
                )
                cv_record.loc[bool_loc, [x == 'opt_params' for x in cv_record.columns]] = str(opt_params)
                cv_record.loc[bool_loc, [x == f'crossval_{self.scoring_metric}' for x in cv_record.columns]] = opt_score
                pf_dict = self.evaluate_model(clf, F_test, y_test)
                for pf_key in pf_dict:
                    cv_record.loc[bool_loc, [x == pf_key for x in cv_record.columns]] = pf_dict[pf_key]
        cv_record.to_csv(outfn, header=True, index=True)
        return
