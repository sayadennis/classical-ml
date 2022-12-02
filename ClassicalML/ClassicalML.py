import os
import sys
import numpy as np
import pandas as pd

# classifiers 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Lasso # using LR so no need to load 
# from sklearn.linear_model import ElasticNet # using LR so no need to load 
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
    def __init__(self, scoring_metric='accuracy', nmf=False, standardscale=True):
        super(ClassicalML).__init__()
        self.scoring_metric = scoring_metric
        self.nmf = nmf
        self.standardscale = standardscale
        if (self.nmf & self.standardscale):
            raise ValueError('Cannot have both nmf=True and standardscale=True because ' \
                             'standard-scaling results in negative values.')
        self.lrm_params = {
            'classifier__C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
        }
        self.lasso_params = {
            'classifier__C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
        }
        self.elasticnet_params = {
            'classifier__C' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3],
            'classifier__l1_ratio' : np.arange(0.1, 1., step=0.1)
        }
        self.svm_params = {
            'classifier__kernel' : ['linear', 'rbf'], 
            'classifier__C' : [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3],
            'classifier__gamma' : [1e-3, 1e-2, 1e-1, 1., 1e+1, 1e+2, 1e+3]
        }
        self.rf_params = {
            # 'criterion' : ['gini', 'entropy'],
            'classifier__max_depth' : [3, 5, 10, 25, 50], # or could set min_samples_split 
            'classifier__min_samples_leaf' : [2, 4, 6, 8, 10, 15, 20]
        }
        self.gb_params = {
            'classifier__loss' : ['deviance', 'exponential'],
            'classifier__min_samples_split' : [2, 6, 10, 15, 20],
            'classifier__max_depth' : [5, 10, 25, 50, 75]
        }
        self.xgb_params = {
            'classifier__learning_rate' : [0.01, 0.1, 0.2],
            'classifier__max_depth' : [5, 10, 15, 25],
            'classifier__colsample_bytree' : [0.3, 0.5, 0.7, 0.9],
            'classifier__gamma' : [0.1, 1, 2]
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
        y_train = np.array(y_train).ravel(); y_test = np.array(y_test).ravel()
        tuner_dict = {
            'LRM' : self.lrm_cv,
            'SVM' : self.svm_cv,
            'LASSO' : self.lasso_cv,
            'ElasticNet' : self.elasticnet_cv,
            'RF' : self.rf_cv,
            'GB' : self.gb_cv,
            'XGB' : self.xgb_cv,
        }
        cv_record = pd.DataFrame(
            None, index=tuner_dict.keys(), 
            columns=['opt_params', f'crossval_{self.scoring_metric}', 'test_bal_acc', 'roc_auc', 'precision', 'recall', 'f1']
        )
        trained_models = {}
        for model_name in tuner_dict.keys():
            opt_params, opt_score, clf = tuner_dict[model_name](X_train, y_train)
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

    # Logistic regression 
    def lrm_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=3000, random_state=seed)
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsLR = GridSearchCV(
                pipe,
                param_grid=self.lrm_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsLR.fit(X_train, y_train)
            opt_params = gsLR.best_params_
            opt_mean_score = np.mean(
                gsLR.cv_results_['mean_test_score'][
                    (gsLR.cv_results_['param_classifier__C'] == opt_params['classifier__C'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=3000, random_state=seed)
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsLR = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.lrm_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsLR.fit(X_train, y_train)
            opt_params = gsLR.best_params_
            opt_mean_score = np.mean(
                gsLR.cv_results_['mean_test_score'][
                    (gsLR.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsLR.cv_results_['param_classifier__C'] == opt_params['classifier__C'])
                ]
            )
        return opt_params, opt_mean_score, gsLR.best_estimator_

    # LASSO cross-validation
    def lasso_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=seed)
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsLS = GridSearchCV(
                pipe,
                param_grid=self.lasso_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsLS.fit(X_train, y_train)
            opt_params = gsLS.best_params_
            opt_mean_score = np.mean(
                gsLS.cv_results_['mean_test_score'][
                    (gsLS.cv_results_['param_classifier__C'] == opt_params['classifier__C'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=seed)
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsLS = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.lasso_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsLS.fit(X_train, y_train)
            opt_params = gsLS.best_params_
            opt_mean_score = np.mean(
                gsLS.cv_results_['mean_test_score'][
                    (gsLS.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsLS.cv_results_['param_classifier__C'] == opt_params['classifier__C'])
                ]
            )
        return opt_params, opt_mean_score, gsLS.best_estimator_

    # Elastic Net cross-validation 
    def elasticnet_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=3000, random_state=seed)
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsEN = GridSearchCV(
                pipe,
                param_grid=self.elasticnet_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsEN.fit(X_train, y_train)
            opt_params = gsEN.best_params_
            opt_mean_score = np.mean(
                gsEN.cv_results_['mean_test_score'][
                    (gsEN.cv_results_['param_classifier__C'] == opt_params['classifier__C']) &
                    (gsEN.cv_results_['param_classifier__l1_ratio'] == opt_params['classifier__l1_ratio'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=3000, random_state=seed)
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsEN = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.elasticnet_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsEN.fit(X_train, y_train)
            opt_params = gsEN.best_params_
            opt_mean_score = np.mean(
                gsEN.cv_results_['mean_test_score'][
                    (gsEN.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsEN.cv_results_['param_classifier__C'] == opt_params['classifier__C']) &
                    (gsEN.cv_results_['param_classifier__l1_ratio'] == opt_params['classifier__l1_ratio'])
                ]
            )
        return opt_params, opt_mean_score, gsEN.best_estimator_

    # Support vector machine cross-validation 
    def svm_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = SVC(class_weight='balanced', max_iter=3000, probability=True, random_state=seed) #, decision_function_shape='ovr'
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsCV = GridSearchCV(
                pipe,
                param_grid=self.svm_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsCV.fit(X_train, y_train)
            opt_params = gsCV.best_params_
            opt_mean_score = np.mean(
                gsCV.cv_results_['mean_test_score'][
                    (gsCV.cv_results_['param_classifier__kernel'] == opt_params['classifier__kernel']) & 
                    (gsCV.cv_results_['param_classifier__C'] == opt_params['classifier__C']) & 
                    (gsCV.cv_results_['param_classifier__gamma'] == opt_params['classifier__gamma'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = SVC(class_weight='balanced', max_iter=3000, probability=True, random_state=seed) #, decision_function_shape='ovr'
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsCV = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.svm_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsCV.fit(X_train, y_train)
            opt_params = gsCV.best_params_
            opt_mean_score = np.mean(
                gsCV.cv_results_['mean_test_score'][
                    (gsCV.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsCV.cv_results_['param_classifier__kernel'] == opt_params['classifier__kernel']) & 
                    (gsCV.cv_results_['param_classifier__C'] == opt_params['classifier__C']) & 
                    (gsCV.cv_results_['param_classifier__gamma'] == opt_params['classifier__gamma'])
                ]
            )
        return opt_params, opt_mean_score, gsCV.best_estimator_

    # Random Forest cross-validation
    def rf_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = RandomForestClassifier(n_estimators=300, criterion='gini', max_features='auto', class_weight='balanced', n_jobs=8, random_state=seed)
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsRF = GridSearchCV(
                pipe,
                param_grid=self.rf_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsRF.fit(X_train, y_train)
            opt_params = gsRF.best_params_
            opt_mean_score = np.mean(
                gsRF.cv_results_['mean_test_score'][
                    (gsRF.cv_results_['param_classifier__max_depth'] == opt_params['classifier__max_depth']) &
                    (gsRF.cv_results_['param_classifier__min_samples_leaf'] == opt_params['classifier__min_samples_leaf'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = RandomForestClassifier(n_estimators=300, criterion='gini', max_features='auto', class_weight='balanced', n_jobs=8, random_state=seed)
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsRF = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.rf_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsRF.fit(X_train, y_train)
            opt_params = gsRF.best_params_
            opt_mean_score = np.mean(
                gsRF.cv_results_['mean_test_score'][
                    (gsRF.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsRF.cv_results_['param_classifier__max_depth'] == opt_params['classifier__max_depth']) &
                    (gsRF.cv_results_['param_classifier__min_samples_leaf'] == opt_params['classifier__min_samples_leaf'])
                ]
            )
        return opt_params, opt_mean_score, gsRF.best_estimator_

    # Gradient Boosting cross-validation
    def gb_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = GradientBoostingClassifier(subsample=0.8, random_state=seed)
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsGB = GridSearchCV(
                pipe,
                param_grid=self.gb_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsGB.fit(X_train, y_train)
            opt_params = gsGB.best_params_
            opt_mean_score = np.mean(
                gsGB.cv_results_['mean_test_score'][
                    (gsGB.cv_results_['param_classifier__loss'] == opt_params['classifier__loss']) &
                    (gsGB.cv_results_['param_classifier__min_samples_split'] == opt_params['classifier__min_samples_split']) &
                    (gsGB.cv_results_['param_classifier__max_depth'] == opt_params['classifier__max_depth'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = GradientBoostingClassifier(subsample=0.8, random_state=seed)
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsGB = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.gb_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsGB.fit(X_train, y_train)
            opt_params = gsGB.best_params_
            opt_mean_score = np.mean(
                gsGB.cv_results_['mean_test_score'][
                    (gsGB.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsGB.cv_results_['param_classifier__loss'] == opt_params['classifier__loss']) &
                    (gsGB.cv_results_['param_classifier__min_samples_split'] == opt_params['classifier__min_samples_split']) &
                    (gsGB.cv_results_['param_classifier__max_depth'] == opt_params['classifier__max_depth'])
                ]
            )
        return opt_params, opt_mean_score, gsGB.best_estimator_

    # XGBoost
    def xgb_cv(self, X_train, y_train, seed=0):
        if self.standardscale:
            scaler = StandardScaler()
            clf = xgb.XGBClassifier(objective='reg:logistic', subsample=1, reg_alpha=0, reg_lambda=1, n_estimators=300, seed=seed)
            pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
            gsXGB = GridSearchCV(
                pipe,
                param_grid=self.xgb_params, n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsXGB.fit(X_train, y_train)
            opt_params = gsXGB.best_params_
            opt_mean_score = np.mean(
                gsXGB.cv_results_['mean_test_score'][
                    (gsXGB.cv_results_['param_classifier__learning_rate'] == opt_params['classifier__learning_rate']) &
                    (gsXGB.cv_results_['param_classifier__max_depth'] == opt_params['classifier__max_depth']) &
                    (gsXGB.cv_results_['param_classifier__colsample_bytree'] == opt_params['classifier__colsample_bytree']) &
                    (gsXGB.cv_results_['param_classifier__gamma'] == opt_params['classifier__gamma'])
                ]
            )
        elif self.nmf:
            nmf = NMF(init='nndsvd', random_state=24)
            clf = xgb.XGBClassifier(objective='reg:logistic', subsample=1, reg_alpha=0, reg_lambda=1, n_estimators=300, seed=seed)
            pipe = Pipeline([('nmf', nmf), ('classifier', clf)])
            gsXGB = GridSearchCV(
                pipe,
                param_grid=(self.nmf_params | self.xgb_params), n_jobs=8, scoring=self.scoring_metric, refit=True,
                cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            )
            gsXGB.fit(X_train, y_train)
            opt_params = gsXGB.best_params_
            opt_mean_score = np.mean(
                gsXGB.cv_results_['mean_test_score'][
                    (gsXGB.cv_results_['param_nmf__n_components'] == opt_params['nmf__n_components']) & 
                    (gsXGB.cv_results_['param_classifier__learning_rate'] == opt_params['classifier__learning_rate']) &
                    (gsXGB.cv_results_['param_classifier__max_depth'] == opt_params['classifier__max_depth']) &
                    (gsXGB.cv_results_['param_classifier__colsample_bytree'] == opt_params['classifier__colsample_bytree']) &
                    (gsXGB.cv_results_['param_classifier__gamma'] == opt_params['classifier__gamma'])
                ]
            )
        # xgb_clf = xgb.XGBClassifier(
        #     objective='reg:logistic', subsample=1, reg_alpha=0, reg_lambda=1, n_estimators=300, seed=seed,
        #     learning_rate=opt_params['classifier__learning_rate'], max_depth=opt_params['classifier__max_depth'],
        #     colsample_bytree=opt_params['classifier__colsample_bytree'], gamma=opt_params['classifier__gamma']
        # )
        # pipe = Pipeline([('scaler', scaler), ('classifier', clf)])
        # pipe.fit(X_train, y_train)
        return opt_params, opt_mean_score, gsXGB.best_estimator_ # pipe

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
