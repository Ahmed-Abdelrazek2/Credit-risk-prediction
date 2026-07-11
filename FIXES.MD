# Fixes



- Moved imputation to happen after the train/test split.
- Computed missing-value medians from the training split only.
- Applied the same imputation values to the test split without letting the model learn from the full dataset.
- Saved two cleaned files instead of one:
  - `clean_cs_training.csv`
  - `clean_cs_test.csv`
- Updated the modeling notebook to load the two saved splits directly instead of splitting the data again.
- Kept feature engineering applied separately to the train and test splits.



The previous flow calculated imputation statistics from the full dataset before splitting. That allowed information from the test set to leak into preprocessing. The new flow keeps the test set separate and gives a more honest evaluation of the model.



- `notebooks/data_prep.ipynb`
- `Credit-risk-prediction-main/notebooks/data_prep.ipynb`
- `Credit-risk-prediction-main/notebooks/Modling.ipynb`

## Result

- Train set: 80%
- Test set: 20%
- Imputation statistics: train only
- Model evaluation: on held-out test data

## Hyperparameter Algorithm

- The tuning algorithm is **RandomizedSearchCV**.
- It tests different combinations of model parameters using cross-validation and selects the best one based on average precision.
- I used it because it is faster than full grid search and works well when the parameter space is large.
- It was applied to both Random Forest and XGBoost in the modeling notebook.

