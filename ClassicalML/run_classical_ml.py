import os
import sys
import getopt
import datetime
import numpy as np
import pandas as pd
import pickle
from ClassicalML import ClassicalML

opts, extraparams = getopt.getopt(sys.argv[1:], 'i:l:o:d:s:n:v:', 
                                  ['input=', 'label=', 'outfn=', 'indexdir=', 'scoring=', 'nmf=', 'savemodel='])

scoring = 'accuracy' # default accuracy score
nmf = False
save_model = False

for o,p in opts:
    if o in ['-i', '--input']:
        inputfn = p
    if o in ['-l', '--label']:
        labfn = p
    if o in ['-o', '--outfn']:
        outfn = p
    if o in ['-d', '--indexdir']:
        indexdir = p
    if o in ['-s', '--scoring']:
        scoring = p
    if o in ['-n', '--nmf']:
        nmf = True
    if o in ['-v', '--savemodel']:
        save_model = True
        modeldir = p

X = pd.read_csv(inputfn, index_col=0)
y = pd.read_csv(labfn, index_col=0)

train_ix = pd.read_csv(f'{indexdir}/train_ix.csv', header=None).to_numpy().ravel()
test_ix = pd.read_csv(f'{indexdir}/test_ix.csv', header=None).to_numpy().ravel()

# split based on indices 
# X_train, X_test = X.iloc[[i in train_ix for i in X.index],:], X.iloc[[i in test_ix for i in X.index],:]
# y_train, y_test = y.iloc[[i in train_ix for i in y.index],:], y.iloc[[i in test_ix for i in y.index],:]
X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
y_train, y_test = y.iloc[train_ix,:], y.iloc[test_ix,:]
# align order
y_train = y_train.loc[X_train.index,:]
y_test = y_test.loc[X_test.index,:]
# print shapes 
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

m = ClassicalML(scoring_metric=scoring)

if nmf: # tune ML with NMF 
    m.record_tuning_NMF(X_train, y_train, X_test, y_test, outfn=outfn, multiclass=False) # k_list=[20,40,60], 
else: # tune regular ML without NMF
    m.record_tuning(X_train, y_train, X_test, y_test, outfn=outfn, multiclass=False)

if save_model:
    inputname = inputfn.split('/')[-1].split('.')[0]
    best_modelname = m.best_model_name
    best_model = m.best_model
    now = datetime.datetime.now()
    date = now.strftime('%Y%m%d')
    if nmf:
        factorized_mx = m.factorized
        with open(f'{modeldir}/{date}_saved_best_nmf_{best_modelname}_{inputname}.p', 'wb') as f:
            pickle.dump(best_model, f)
        with open(f'{modeldir}/{date}_saved_factorized_mx_for_{best_modelname}_{inputname}.p', 'wb') as f:
            pickle.dump(factorized_mx, f)
    else: # not NMF 
        with open(f'{modeldir}/{date}_saved_best_{best_modelname}_{inputname}.p', 'wb') as f:
            pickle.dump(best_model, f)
