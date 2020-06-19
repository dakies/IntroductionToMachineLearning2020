# 'Task 2 IML'

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import preprocessing

import pickle

"""
    -- IML Project Task 2 Code by Daniel Kiesewalter and Leiv Andresen --
    - execution requirements below -

    - Task description:

    The task is split into three subtasks which all deal to reason about the relative taste of food on images. More precisely we are presented
    with three images of food, A, B and C, and we need to say if the food in image A tastes more like the
    food in image B or C. 

    - Approach description: 

    - Execution requirements:

    Tested only on Python version 3.7
    modules: sklearn, pandas, numpy
    pickle is used to save/load objects: https://docs.python.org/3/library/pickle.html
    For the download of the NasNetLarge model to work we had to follow the first answer here: 
    https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664
    and then install pillow: 
    https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664

    "train_triplets.txt", "test_triplets.txt" and the folder "food" with the images have to be located in the
    same folder as the script. The script will create and save files to the folder "NasNetLarge_features_balanced".

    The total space required is roughly 25 GB and the time required is around 6 hours on our machine.
    The results will be saved to the folder in which the script is placed.
    If the script is rerun then it will load the features from the disk.
    If you have problems feel free to contact us at leiva@ethz.ch.

    """


# Random states
random_state = 123
np.random.seed(random_state)


# Impute Raw Data
def impute(dataframe):
    # Forwardfill per patient
    dataframe.loc[:, dataframe.columns != 'pid'] = dataframe.groupby('pid').transform(lambda x: x.ffill())
    # Backwardfill per patient
    dataframe.loc[:, dataframe.columns != 'pid'] = dataframe.groupby('pid').transform(lambda x: x.bfill())

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

    print("Preprocessing data finished.")

# Split into test and train
y = train_labels
x_train = grouped
y_train = y

# Support vector machine with Gridsearch
labels_clf = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2', 'LABEL_Sepsis']

# Parameters to search over
# {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf']},
param_grid = [
   {'randomforestclassifier__min_samples_split': [2, 3, 4, 8],
    'randomforestclassifier__min_samples_leaf': [4, 10, 20],
    'simpleimputer__strategy': ['mean', 'median', 'most_frequent']}
]
# Use Area Under the Receiver Operating Characteristic Curve (ROC AUC) as performance metric in Gridsearch
score = 'roc_auc'

# Initialize dataframe for results
results = pd.DataFrame()
results['pid'] = test_features['pid'].unique()

results_train_data = pd.DataFrame()
results_train_data['pid'] = train_data['pid'].unique()
all_scores = {}

print('SVM start')
for index, label in enumerate(labels_clf):
    estimator = make_pipeline(
        SimpleImputer(),
        preprocessing.StandardScaler(),
        SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, random_state=random_state)),
        RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state, class_weight='balanced')
    )

    print("# Tuning hyper-parameters for label %s" % (label))

    clf = GridSearchCV(
        estimator, param_grid, n_jobs=-1, scoring=score, cv=2
    )
    clf.fit(x_train, y_train.loc[:, label].values)

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

    results[label] = clf.predict_proba(grouped_t)[:, 1]

    all_scores[("ROC_" + label)] = clf.best_score_
    print("ROC Score: ", all_scores[("ROC_" + label)])

    results[label] = clf.predict_proba(grouped_t)[:, 1]
    results_train_data[label] = clf.predict_proba(grouped)[:, 1]


# Regression for prediction of future values
print('Regression start')
labels_reg = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

param_grid_reg = [{'randomforestregressor__min_samples_split': [2, 4, 8],
                       'randomforestregressor__min_samples_leaf': [4, 10, 20],
                       'simpleimputer__strategy': ['mean', 'median', 'most_frequent']}
                  ]

score = 'r2'
for index, label in enumerate(labels_reg):
    estimator = make_pipeline(
        SimpleImputer(),
        preprocessing.StandardScaler(),
        SelectFromModel(linear_model.Lasso(alpha=0.01)),
        RandomForestRegressor(n_estimators=100, random_state=random_state)
    )
    print("# Tuning hyper-parameters for label %s" % label)

    reg = GridSearchCV(
        estimator, param_grid_reg, scoring=score, n_jobs=-1, cv=2
    )
    reg.fit(x_train, y_train.loc[:, label])

    print("Best parameters set found on development set:")
    print()
    print(reg.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = reg.cv_results_['mean_test_score']
    stds = reg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print()
    results[label] = reg.predict(grouped_t)

    results_train_data[label] = reg.predict(grouped)

    all_scores[("R2_" + label)] = reg.best_score_
    print("R2 Score: ", all_scores[("R2_" + label)])

print('Regression end')

# print results for spreadsheet
for label in labels_clf:
    print(all_scores[("ROC_" + label)])
for label_2 in labels_reg:
    print(all_scores[("R2_" + label_2)])

# Save Results
results.to_csv('prediction_reproduce_best_all_data_train_val_final.zip', index=False, float_format='%.3f', compression='zip')

print('Results saved as prediction.zip')
