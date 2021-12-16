import os
import sys
import getopt
import datetime
import numpy as np
import pandas as pd
import pickle

sys.path.append('classical_ml')
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

X = pd.read_csv(inputfn, index_col=0)
y = pd.read_csv(labfn, index_col=0)

train_ix = pd.read_csv('%s/train_ix.csv' % (indexdir), header=None)
test_ix = pd.read_csv('%s/test_ix.csv' % (indexdir), header=None)

X_train, X_test = X.iloc[train_ix[0]], X.iloc[test_ix[0]]
y_train, y_test = y.iloc[train_ix[0]], y.iloc[test_ix[0]]

m = ClassicalML(scoring_metric=scoring)

if nmf: # tune ML with NMF 
    m.record_tuning_NMF(X_train, y_train, X_test, y_test, outfn=outfn, k_list=[20,40,60], multiclass=False)
else: # tune regular ML without NMF
    m.record_tuning(X_train, y_train, X_test, y_test, outfn=outfn, multiclass=False)


if save_model:
    inputname = inputfn.split('/')[-1].split('.')[0]
    best_modelname = m.best_model_name
    best_model = m.best_model
    now = datetime.datetime.now()
    date = now.strftime('%Y%m%d')
    with open(f'{date}_saved_best_{best_modelname}_{inputname}.p', 'wb') as f:
        pickle.dump(best_model, f)
