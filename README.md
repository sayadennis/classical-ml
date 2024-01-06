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
python classical-ml/ClassicalML/run_classical_ml.py \
    --input path/to/read/input.csv \
    --label path/to/read/target.csv \
    --outdir path/to/directory/to/write/outputs/ \
    --indexdir path/to/read/indices/ # index files should be named train_index.txt and test_index.txt \
    --scoring roc_auc \
    --nmf 500;
```

### Data setup 

* Input -- CSV with a header row and an index column 
* Target -- CSV with a header row and an index column with shape `(-1,1)`. Index must be consistent with input. 
* Train-test indices -- TXT file with train/test indices. Must be consistent with input and target indices. 
