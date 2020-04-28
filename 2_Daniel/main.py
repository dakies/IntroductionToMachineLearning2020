# 'Task 2 IML'

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

# Load data
# test_data = pd.read_csv("test_features.csv", index_col="pid")
train_data = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")
test_features = pd.read_csv("test_features.csv")

# Check which coloumns require imputation
num_var = train_data.shape[1]
for i in range(num_var):
    print(f'Column {train_data.columns[i]} contains: {train_data[train_data.columns[i]].isna().sum()} Nans')
# Determine all patient id's
pids = pd.unique(train_data["pid"])
# Split into patients into Train and Validation??? Why not k-fold CV?
pids_train, pids_val = train_test_split(pids)

# # Determine statistical values for imputation
# train_mean = train_data.loc[train_data["pid"].isin(pids_train)].mean().values
# train_median = train_data.loc[train_data["pid"].isin(pids_train)].median().values
# train_mode = train_data.loc[train_data["pid"].isin(pids_train)].mode().values

# Create training dataset including pid, time age + following features
# features = ['max', 'min', 'mean', 'median', 'mode', 'std']
features = []
# Empty array for variable statistics to append to
var_stat = np.empty([len(pids_train), 3 + (37-3) * features.__len__()])
counter = 0
print('Start imputation of training data')
for pid in pids_train:
    # Select time series for one patient
    X_pid = train_data.loc[train_data["pid"] == pid].values
    # Put Pid, time offset and age in first 2 coloumns
    var_stat_pid = X_pid[1, [0, 1, 2]]
    # Select all variables other than pid time and age
    X_pid = X_pid[:, 3:]
    if 'max' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmax(X_pid, axis=0))
    if 'min' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmin(X_pid, axis=0))
    if 'mean' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmean(X_pid, axis=0))
    if 'median' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmedian(X_pid, axis=0))
    if 'mode' in features:
        mode, count = stats.mode(X_pid, axis=0, nan_policy='omit')
        var_stat_pid = np.append(var_stat_pid, mode.data)
    if 'std' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanstd(X_pid, axis=0))
    var_stat[counter, :] = var_stat_pid
    counter += 1
x_train = var_stat
# Average fpr imputation
var_stat = np.nanmean(var_stat, axis=0)
# Todo Try other methods than mean
num_feat = range(var_stat.shape[0])
# Impute
for col in num_feat:
    x_train[:, col] = np.nan_to_num(x_train[:, col], nan=var_stat[col])

# Create validation dataset labels
# Empty array for variable statistics to append to
var_stat = np.empty([len(pids_val), 3 + (37 - 3) * features.__len__()])
counter = 0
print('Start imputation of validation data')
for pid in pids_val:
    # Select time series for one patient
    X_pid = train_data.loc[train_data["pid"] == pid].values
    # Put Pid anf age in first 2 coloumns
    var_stat_pid = X_pid[1, [0, 1, 2]]
    # Select all variables other than pid time and age
    X_pid = X_pid[:, 3:]
    if 'max' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmax(X_pid, axis=0))
    if 'min' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmin(X_pid, axis=0))
    if 'mean' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmean(X_pid, axis=0))
    if 'median' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmedian(X_pid, axis=0))
    if 'mode' in features:
        mode, count = stats.mode(X_pid, axis=0, nan_policy='omit')
        var_stat_pid = np.append(var_stat_pid, mode.data)
    if 'std' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanstd(X_pid, axis=0))
    var_stat[counter, :] = var_stat_pid
    counter += 1
# Todo Check var_stat for correctness
x_val = var_stat
# Average fpr imputation
var_stat = np.nanmean(var_stat, axis=0)
num_feat = range(var_stat.shape[0])
# Impute
for col in num_feat:
    x_val[:, col] = np.nan_to_num(x_val[:, col], nan=var_stat[col])

# Create test dataset labels
# Determine all patient id's
pids = pd.unique(test_features["pid"])
# Empty array for variable statistics to append to
var_stat = np.empty([len(pids), 3 + (37 - 3) * features.__len__()])
counter = 0
print('Start imputation of ''test'' data')
for pid in pids:
    # Select time series for one patient
    X_pid = test_features.loc[test_features["pid"] == pid].values
    # Put Pid anf age in first 2 coloumns
    var_stat_pid = X_pid[1, [0, 1, 2]]
    # Select all variables other than pid time and age
    X_pid = X_pid[:, 3:]
    if 'max' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmax(X_pid, axis=0))
    if 'min' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmin(X_pid, axis=0))
    if 'mean' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmean(X_pid, axis=0))
    if 'median' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanmedian(X_pid, axis=0))
    if 'mode' in features:
        mode, count = stats.mode(X_pid, axis=0, nan_policy='omit')
        var_stat_pid = np.append(var_stat_pid, mode.data)
    if 'std' in features:
        var_stat_pid = np.append(var_stat_pid, np.nanstd(X_pid, axis=0))
    var_stat[counter, :] = var_stat_pid
    counter += 1
x_test = var_stat
# Average for imputation
var_stat = np.nanmean(var_stat, axis=0)
# Todo Try other methods than mean
num_feat = range(var_stat.shape[0])
# Impute
for col in num_feat:
    x_test[:, col] = np.nan_to_num(x_test[:, col], nan=var_stat[col])

# Select training patients from labels
y_train = train_labels.loc[train_labels['pid'].isin(pids_train)]
y_val = train_labels.loc[train_labels['pid'].isin(pids_val)]

#Check no Nan included in train data
for i in range(num_var):
    print(f'X_train column {i} contains Nans: {np.isnan(x_train).any()} ')

# Support vector machine with Gridsearch
labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2', 'LABEL_Sepsis']

# Parameters to search over
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
# Use Area Under the Receiver Operating Characteristic Curve (ROC AUC) as performance metric in gridsearch
score = 'roc_auc'
print('SVM start')
for index, label in enumerate(labels):
    print("# Tuning hyper-parameters for %s for label %s" % (score, label))

    clf = GridSearchCV(
        SVC(probability=False), param_grid, scoring=score, n_jobs=-1, cv=2
    )
    clf.fit(x_train[:, 1:], y_train.loc[:, label])

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val.loc[:, label], clf.predict(x_val[:, 1:])
    print(classification_report(y_true, y_pred))
    print()

# Lasso for prediction of future values
print('Lasso start')
labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
alphas_lasso = [{'alpha': [0.01, 0.1, 1, 10, 100]}]
score = 'r2'
for index, label in enumerate(labels):
    print("# Tuning hyper-parameters for %s for label %s" % (score, label))

    reg = GridSearchCV(
        linear_model.Lasso(), alphas_lasso, scoring=score, n_jobs=-1
    )
    reg.fit(x_train[:, 1:], y_train.loc[:, label])

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val.loc[:, label], clf.predict(x_val[:, 1:])
    print(r2_score(y_true, y_pred))
    print()
print('Lasso end')

# for pid in pids_train:
#     # Get training data for patient
#     X_pid = train_data.loc[train_data["pid"] == pid].values
#     imputed_time_series = {}  # python sets
#     for i in range(num_var):
#         if(np.isnan(X_pid[:, i].max())):
#             #= np.nan_to_num(X_pid[:, i], nan=train_mean[0, i])
#             imputed_time_series['median', i] = np.nan_to_num(X_pid[:, i], nan=train_median[0, i])
#             # = np.nan_to_num(X_pid[:, i], nan=train_mode[0, i])
#         else:
#             X_pid[:, i].max()

#     # Construct features of choice
#     feat_row = []
#     for feat_desc in FEAT_DESCS:
#         needed_inputs, func = feat_desc
#         args = []
#
#
#         for var, require_impute, strategy in needed_inputs:
#             if require_impute:
#                 input = imputed_time_series[var]
#             else:
#                 input = train_data.loc[train_data["pid"] == pid][var]
#
#             args.append(input)
#         feature = func(args)
#         feat_row.append(feature)
#     X_all.append(feat_row)
# X_train = np.concatenate(X_all, axis=0)
