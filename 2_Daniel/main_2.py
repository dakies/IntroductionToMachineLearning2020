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
from sklearn import preprocessing

# Load data
# test_data = pd.read_csv("test_features.csv", index_col="pid")
train_data = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")
test_features = pd.read_csv("test_features.csv")

grouped = train_data.groupby(['pid'], sort=False).agg([np.mean, np.min, np.max, np.std])
grouped_t = test_features.groupby(['pid'], sort=False).agg([np.mean, np.min, np.max, np.std])

#Remove all patients with no data apart from age and time
grouped = grouped.dropna(thresh=8)

#Split into test and train
y = train_labels
x_train, x_test, y_train, y_test = train_test_split(grouped, y, test_size=0.1)

#Impute with mode
imputer = SimpleImputer(strategy="most_frequent")
x_train = imputer.fit_transform(x_train)
x_test = imputer.fit_transform(x_test)
x_t = imputer.fit_transform(grouped_t)

#Scale
x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
x_t = preprocessing.scale(x_t)

# Support vector machine with Gridsearch
labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2', 'LABEL_Sepsis']

# Parameters to search over
#{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']}
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},
]
# Use Area Under the Receiver Operating Characteristic Curve (ROC AUC) as performance metric in Gridsearch
score = 'roc_auc'

#Initialize dataframe for results
results = pd.DataFrame()
results['pid'] = test_features['pid'].unique()

print('SVM start')
for index, label in enumerate(labels):
    print("# Tuning hyper-parameters for %s for label %s" % (score, label))

    clf = GridSearchCV(
        SVC(probability=True, cache_size=1000), param_grid, scoring=score, n_jobs=-1, cv=2
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
    y_true, y_pred = y_test.loc[:, label], clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    results[label] = clf.predict_proba(x_t)[:, 1]


# Lasso for prediction of future values
print('Lasso start')
labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
alphas_lasso = [{'alpha': [0.01, 0.1, 1, 10, 100]}]
score = 'r2'
for index, label in enumerate(labels):
    print("# Tuning hyper-parameters for %s for label %s" % (score, label))

    reg = GridSearchCV(
        linear_model.Lasso(max_iter=10000), alphas_lasso, scoring=score, n_jobs=-1
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
    y_true, y_pred = y_test.loc[:, label], reg.predict(x_test)
    print(r2_score(y_true, y_pred))
    print()
    results[label] = reg.predict(x_t)
print('Lasso end')

#Save Results
results.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

print('Results saved as prediction.zip')