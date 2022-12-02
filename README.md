# Module for Classical Machine Learning Model Training and Evaluation

This is a flexible module that provides baseline classical ML model training and evaluation for any small datasets. 

## What it does 

### Training 

The tool performs hyperparameter tuning & model training via 5-fold cross-validation on the training set, testing 5 models: 

* Logistic Regression
* LASSO
* Elastic Net 
* Support Vector Machine
* Gradient Boosting Tree
* Random Forest
* XGBoost

### Evaluation 

The tool output includes the model artifact and a range of performance metrics on both the train set and test set. 

## Usage 

### Running the CLI

```
python ClassicalML/run_classical_ml.py \
    --input path/to/read/input.csv \
    --label path/to/read/target.csv \
    --outfn path/to/write/performance.csv \
    --indexdir path/to/read/indices/ # index files should be named train_ix.csv and test_ix.csv \
    --scoring roc_auc # sklearn's metric name \
    --nmf 500 # number of max components if performing NMF \
    --savemodel path/to/save/model_artifact.p # will be save as a pickle file
```

### Data setup 

* Input -- `pandas.DataFrame` with index and columns included (index will be used to align matrix with target) 
* Target -- `pandas.DataFrame` with shape `(-1,1)`. Must be consistent with input. 
* Train-test indices -- TXT file with train/test indices. Must be consistent with input and target. 
