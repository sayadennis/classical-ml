import os
import sys
import getopt
import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from ClassicalML import ClassicalML

opts, extraparams = getopt.getopt(
    sys.argv[1:],
    "i:l:o:d:s:n:c:",
    ["input=", "label=", "outdir=", "indexdir=", "scoring=", "nmf=", "n_cpu="],
)

scoring = "accuracy"  # default accuracy score
nmf = False
standardscale = True
save_model = False

for o, p in opts:
    if o in ["-i", "--input"]:
        inputfn = p
    if o in ["-l", "--label"]:
        labfn = p
    if o in ["-o", "--outdir"]:
        outdir = p
    if o in ["-d", "--indexdir"]:
        indexdir = p
    if o in ["-s", "--scoring"]:
        scoring = p
    if o in ["-n", "--nmf"]:
        nmf = True  # max components ignored for now
    if o in ["-c", "--n_cpu"]:
        n_cpu = int(p)

X = pd.read_csv(inputfn, index_col=0)
y = pd.read_csv(labfn, index_col=0)

train_ix = pd.read_csv(f"{indexdir}/train_index.txt", header=None).to_numpy().ravel()
test_ix = pd.read_csv(f"{indexdir}/test_index.txt", header=None).to_numpy().ravel()

# split based on indices
X_train, X_test = X.loc[train_ix, :], X.loc[test_ix, :]
y_train, y_test = y.loc[train_ix, :], y.loc[test_ix, :]

# print shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

inputname = inputfn.split("/")[-1].split(".")[0]
date = datetime.datetime.now().strftime("%Y%m%d")
outfn = f"{outdir}/{date}_performance_{inputname}.csv"

if nmf:
    standardscale = False

m = ClassicalML(
    scoring_metric=scoring, nmf=nmf, standardscale=standardscale, n_cpu=n_cpu
)
m.record_tuning(X_train, y_train, X_test, y_test, outfn=outfn, multiclass=False)

best_modelname = m.best_model_name
best_model = m.best_model
with open(f"{outdir}/{date}_saved_best_{best_modelname}_{inputname}.p", "wb") as f:
    pickle.dump(best_model, f)

########################
#### Plot ROC Curve ####
########################

n_classes = len(y_test.iloc[:, 0].unique())

y_score = best_model.predict_proba(X_test)[:, 1]  # decision_function

if n_classes == 2:  # binary
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # plot
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"{outdir}/{date}_ROC_best_{best_modelname}_{inputname}.png")
    plt.close()
else:  # multiclass
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Pperating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"{outdir}/{date}_ROC_best_{best_modelname}_{inputname}.png")
    plt.close()

###############################
#### Save confucion matrix ####
###############################

y_test_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred)
pd.DataFrame(
    cm, index=["true neg", "true pos"], columns=["pred neg", "pred pos"]
).to_csv(f"{outdir}/{date}_confusion_mx_best_{best_modelname}_{inputname}.csv")
