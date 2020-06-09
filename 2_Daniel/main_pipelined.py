# 'Task 2 IML'

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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

grouped = train_data.groupby(['pid'], sort=False).agg([np.mean, np.min, np.max, np.std, np.first])
grouped_t = test_features.groupby(['pid'], sort=False).agg([np.mean, np.min, np.max, np.std, np.first])

#Remove all patients with no data apart from age and time
grouped = grouped.dropna(thresh=8)

#Split into test and train
y = train_labels
x_train, x_test, y_train, y_test = train_test_split(grouped, y, test_size=0.01)

# Moved this stuff into pipline to prevent leakage of information into validation data during scaling
# #Impute with mode
# imputer = SimpleImputer(strategy="most_frequent")
# x_train = imputer.fit_transform(x_train)
# x_test = imputer.fit_transform(x_test)
# x_t = imputer.fit_transform(grouped_t)
#
# #Scale
# x_train = preprocessing.scale(x_train)
# x_test = preprocessing.scale(x_test)
# x_t = preprocessing.scale(x_t)

# Support vector machine with Gridsearch
labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
       'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
       'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
       'LABEL_EtCO2', 'LABEL_Sepsis']

# Parameters to search over
# {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf']},
param_grid = [
   {'randomforestclassifier__min_samples_split': [2, 4], 'simpleimputer__strategy': ['mean', 'median', 'most_frequent']}
]
# Use Area Under the Receiver Operating Characteristic Curve (ROC AUC) as performance metric in Gridsearch
score = 'roc_auc'

#Initialize dataframe for results
results = pd.DataFrame()
results['pid'] = test_features['pid'].unique()

print('SVM start')
for index, label in enumerate(labels):
    estimator = make_pipeline(
        SimpleImputer(),
        preprocessing.StandardScaler(),
        # SVC(probability=True, cache_size=1000, class_weight='balanced')
        RandomForestClassifier(max_depth=None)
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
    y_true, y_pred = y_test.loc[:, label], clf.predict(x_test)
    print(classification_report(y_true, y_pred))

    results[label] = clf.predict_proba(grouped_t)[:, 1]



# Lasso for prediction of future values
print('Lasso start')
labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
# {'randomforestregressor__min_samples_split': [2, 4, 10]}
alphas_lasso = [{'lasso__alpha': [0.01, 0.1, 1, 10, 100]}]
score = 'r2'
for index, label in enumerate(labels):
    score_simple_imputer_lasso = [-100, 'estimator']
    for strategy in ('mean', 'median', 'most_frequent'):

        estimator = make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy=strategy),
            preprocessing.StandardScaler(),
            # RandForestRegressor
            linear_model.Lasso(max_iter=10000)
        )
        print("# Tuning hyper-parameters for label %s" % (label))

        reg = GridSearchCV(
            estimator , alphas_lasso, scoring=score, n_jobs=-1
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

        if reg.best_score_ > score_simple_imputer_lasso[0]:
            score_simple_imputer_lasso[0] = reg.best_score_
            score_simple_imputer_lasso[1] = reg

    reg = score_simple_imputer_lasso[1]
    results[label] = reg.predict(grouped_t)
print('Lasso end')

#Save Results
results.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

print('Results saved as prediction.zip')