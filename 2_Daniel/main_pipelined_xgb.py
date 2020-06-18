# 'Task 2 IML'

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

import xgboost as gb
import pickle

# Random states
random_state = 123
np.random.seed(random_state)


# Impute Raw Data
def impute(dataframe):
    # stat = dataframe.mean()
    # Forwardfill per patient
    dataframe.loc[:, dataframe.columns != 'pid'] = dataframe.groupby('pid').transform(lambda x: x.ffill())
    # Backwardfill per patient
    dataframe.loc[:, dataframe.columns != 'pid'] = dataframe.groupby('pid').transform(lambda x: x.bfill())

    # for feature in dataframe:
    #    dataframe[feature] = dataframe[feature].fillna(stat[feature])

    return dataframe


# Load data -  try existing otherwise recompute
try:
    train_labels = pd.read_csv("train_labels.csv")
    train_data = pickle.load(open("train_data_imputed.p", 'rb'))
    test_features = pickle.load(open("test_features_imputed.p", 'rb'))
    grouped = pickle.load(open("grouped_aggregated_train_features.p", 'rb'))
    grouped_t = pickle.load(open("grouped_aggregated_test_features.p", 'rb'))
    print("Successfully loaded preprocessed data.")

except FileNotFoundError:
    print("Preprocessing data ...")
    # test_data = pd.read_csv("test_features.csv", index_col="pid")
    train_data = pd.read_csv("train_features.csv")
    train_labels = pd.read_csv("train_labels.csv")
    test_features = pd.read_csv("test_features.csv")

    train_data = impute(train_data)
    test_features = impute(test_features)

    grouped = train_data.groupby(['pid'], sort=False).agg([np.mean, np.min, np.max, np.std, np.var, 'first', 'last'])
    grouped_t = test_features.groupby(['pid'], sort=False).agg(
        [np.mean, np.min, np.max, np.std, np.var, 'first', 'last'])

    pickle.dump(train_data, open("train_data_imputed.p", 'wb'))
    pickle.dump(test_features, open("test_features_imputed.p", 'wb'))
    pickle.dump(grouped, open("grouped_aggregated_train_features.p", 'wb'))
    pickle.dump(grouped_t, open("grouped_aggregated_test_features.p", 'wb'))

    # train_data.to_csv("train_data_imputed.csv")
    # test_features.to_csv("test_data_imputed.csv")
    # grouped.to_csv("grouped_aggregated_training_features.csv")
    # grouped_t.to_csv("grouped_aggregated_test_features.csv")
    print("Preprocessing data finished.")

data_to_use = 'grouped'
if data_to_use is 'grouped':
    # Split into test and train
    y = train_labels
    x_train, x_test, y_train, y_test = train_test_split(grouped, y, test_size=0.1, random_state=random_state)

    # Impute with mode
    print("Imputing with mode:")
    imputer = SimpleImputer(strategy="most_frequent")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.fit_transform(x_test)
    x_t = imputer.fit_transform(grouped_t)

if data_to_use is 'raw':
    # Pivot these snitches
    one_to_twelve = np.tile(np.linspace(1, 12, 12, dtype=int), int(len(train_data) / 12))
    train_data.insert(1, "hour", one_to_twelve, True)
    train_data = train_data.set_index(['pid', 'hour'])
    train_data = train_data.unstack(level=1)
    one_to_twelve = np.tile(np.linspace(1, 12, 12, dtype=int), int(len(test_features) / 12))
    test_features.insert(1, "hour", one_to_twelve, True)
    test_features = test_features.set_index(['pid', 'hour'])
    test_features = test_features.unstack(level=1)

    # Split into test and train
    y = train_labels
    x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size=0.1, random_state=random_state)

    # Impute with mode
    print("Imputing with mode:")
    imputer = SimpleImputer(strategy="most_frequent")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.fit_transform(x_test)
    test_features = imputer.fit_transform(test_features)
#
# #Scale
# x_train = preprocessing.scale(x_train)
# x_test = preprocessing.scale(x_test)
# x_t = preprocessing.scale(x_t)

# Support vector machine with Gridsearch
labels_clf = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
          'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
          'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
          'LABEL_EtCO2', 'LABEL_Sepsis']
labels_reg = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

# XGB ftw
print("Starting XGB Classification...")
models_clf = {"label": 'model'}
for label in labels_clf:
    y_train_xgb = y_train[label]
    y_test_xgb = y_test[label]
    # Compute class weights -- DOESN'T do anything
    # class_weights = compute_class_weight('balanced', np.unique(y_train_xgb), y_train_xgb)

    xgb_clf = gb.XGBClassifier(random_state=random_state) # class_weights=class_weights)
    xgb_clf.fit(x_train, y_train_xgb, verbose=True)
    y_pred = xgb_clf.predict(x_test)
    print(f"ROC score for label {label}: {'%.4f' % (roc_auc_score(y_true=y_test_xgb, y_score=y_pred))}")
    models_clf[label] = xgb_clf

print("Starting XGB Regression...")
models_reg = {"label": 'model'}
for label in labels_reg:
    y_train_xgb = y_train[label]
    y_test_xgb = y_test[label]
    # Compute class weights -- DOESN'T do anything
    # class_weights = compute_class_weight('balanced', np.unique(y_train_xgb), y_train_xgb)

    xgb_reg = gb.XGBRegressor(random_state=random_state) # class_weights=class_weights)
    xgb_reg.fit(x_train, y_train_xgb, verbose=True)
    y_pred = xgb_reg.predict(x_test)
    print(f"R2 score for label {label}: {'%.4f' % (r2_score(y_true=y_test_xgb, y_pred=y_pred))}")
    models_reg[label] = xgb_reg

# Create evaluation file
results = pd.DataFrame()
results['pid'] = test_features['pid'].unique()
for label in labels_clf:
    results[label] = models_clf[label].predict(x_t)
for label in labels_reg:
    results[label] = models_reg[label].predict(x_t)

results.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

#
# # Parameters to search over
# # {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf']},
# param_grid = [
#    {'randomforestclassifier__min_samples_split': [2, 4, 8],
#     'randomforestclassifier__min_samples_leaf': [1, 2, 4]
#     }
# ]
# # Use Area Under the Receiver Operating Characteristic Curve (ROC AUC) as performance metric in Gridsearch
# score = 'roc_auc'
#
# #Initialize dataframe for results
# results = pd.DataFrame()
# results['pid'] = test_features['pid'].unique()
#
# print('SVM start')
# for index, label in enumerate(labels):
#     estimator = make_pipeline(
#         SimpleImputer(),
#         preprocessing.StandardScaler(),
#         # SVC(probability=True, cache_size=1000, class_weight='balanced')
#         # SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False)),
#         RandomForestClassifier(max_depth=None, class_weight='balanced')
#     )
#
#     print("# Tuning hyper-parameters for label %s" % (label))
#
#     clf = GridSearchCV(
#         estimator, param_grid, n_jobs=-1, scoring=score, cv=2
#     )
#     clf.fit(x_train, y_train.loc[:, label].values)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test.loc[:, label], clf.predict(x_test)
#     print(classification_report(y_true, y_pred))
#
#     results[label] = clf.predict_proba(grouped_t)[:, 1]
#
#
# # Regression for prediction of future values
# print('Regression start')
# labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
#
# param_grid_reg = [
#     {'randomforestregressor__min_samples_split': [2, 4, 8],
#      'randomforestregressor__min_samples_leaf': [1, 2, 4],
#      'simpleimputer__strategy': ['mean', 'median', 'most_frequent']}
#  ]
#
# # param_grid_reg = [
# #    {'lasso__alpha': [0.1, 1, 10, 100],
# #    randomforestregressor__min_samples_split': [2, 4, 8],
# #     'randomforestregressor__min_samples_leaf': [1, 2, 4],
# #     'simpleimputer__strategy': ['mean', 'median', 'most_frequent']}
# # ]
#
# score = 'r2'
# for index, label in enumerate(labels):
#     estimator = make_pipeline(
#         SimpleImputer(),
#         preprocessing.StandardScaler(),
#         # RandomForestRegressor()
#         SelectFromModel(linear_model.Lasso(alpha=0.01)),
#         RandomForestRegressor()
#     )
#     print("# Tuning hyper-parameters for label %s" % label)
#
#     reg = GridSearchCV(
#         estimator, param_grid_reg, scoring=score, n_jobs=-1
#     )
#     reg.fit(x_train, y_train.loc[:, label])
#
#     print("Best parameters set found on development set:")
#     print()
#     print(reg.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = reg.cv_results_['mean_test_score']
#     stds = reg.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, reg.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test.loc[:, label], reg.predict(x_test)
#     print(r2_score(y_true, y_pred))
#     print()
#     results[label] = reg.predict(grouped_t)
# print('Lasso end')
#
# # Save Results
# results.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')
#
# print('Results saved as prediction.zip')
