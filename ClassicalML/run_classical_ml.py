import os
import sys
import getopt
import datetime
import numpy as np
import pandas as pd

sys.path.append('classical_ml')
from ClassicalML import ClassicalML

opts, extraparams = getopt.getopt(sys.argv[1:], 'i:l:o:i:', 
                                  ['input=', 'label=', 'outfn=', 'indexdir='])

for o,p in opts:
    if o in ['-i', '--input']:
        inputfn = p
    if o in ['-l', '--label']:
        labfn = p
    if o in ['-o', '--outfn']:
        outfn = p
    if o in ['-i', '--indexdir']:
        indexdir = p

X = pd.read_csv(inputfn, index_col=0)
y = pd.read_csv(labfn, index_col=0)

train_ix = pd.read_csv('%s/train_ix.csv' % (indexdir), header=None)
test_ix = pd.read_csv('%s/test_ix.csv' % (indexdir), header=None)

X_train, X_test = X.iloc[train_ix[0]], X.iloc[test_ix[0]]
y_train, y_test = y.iloc[train_ix[0]], y.iloc[test_ix[0]]

m = ClassicalML()
m.record_tuning(X_train, y_train, X_test, y_test, outfn=outfn, multiclass=False)
