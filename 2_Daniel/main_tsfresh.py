# 'Task 2 IML'

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.base import BaseEstimator, TransformerMixin

# Impute Raw Class
# Wanted to put this step in pipeline, but woul dbe too muc work as we would have to design container for y
class ImputeRaw( BaseEstimator, TransformerMixin ):
    """Parameters
    strategy : string, default='mean'
        The imputation strategy."""
    # Class Constructor
    def __init__(self, strategy="mean"):
        self.strategy = strategy

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.strategy == "mean":
            stat = X.mean()
        elif self.strategy == "median":
            stat = X.median()
        elif self.strategy == "most_frequent":
            stat = X.mode()[0]
        else:
            allowed_strategies = ["mean", "median", "most_frequent", "constant"]
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))
        # Forward-fill per patient
        X.loc[:, X.columns != 'pid'] = X.groupby('pid').transform(lambda x: x.ffill())
        # Backward-fill per patient
        X.loc[:, X.columns != 'pid'] = X.groupby('pid').transform(lambda x: x.bfill())

        for feature in X:
            X[feature] = X[feature].fillna(stat[feature])
        return X


# Impute Raw Data
def raw_impute(dataframe):
    stat = dataframe.mean()
    # Forwardfill per patient
    dataframe.loc[:, dataframe.columns != 'pid'] = dataframe.groupby('pid').transform(lambda x: x.ffill())
    # Backwardfill per patient
    dataframe.loc[:, dataframe.columns != 'pid'] = dataframe.groupby('pid').transform(lambda x: x.bfill())

    for feature in dataframe:
        dataframe[feature] = dataframe[feature].fillna(stat[feature])

    return dataframe

# Load data
# raw: contains Nan
x_raw = pd.read_csv("train_features.csv")
y = pd.read_csv("train_labels.csv")
x_test_raw = pd.read_csv("test_features.csv")


# Classification Labels
labels_class = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
          'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
          'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
          'LABEL_EtCO2', 'LABEL_Sepsis']

# Regression Labels
labels_reg = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

# Train test splitting
pids_train, pids_val = train_test_split(y["pid"], test_size=0.99)
x_raw = x_raw.loc[x_raw["pid"].isin(pids_train)]
x_val = x_raw.loc[x_raw["pid"].isin(pids_val)]
y = y.loc[y["pid"].isin(pids_train)]
y_val = y.loc[y["pid"].isin(pids_val)]
# Preprocess x
print("Raw imputation...")
x = raw_impute(x_raw)
x_test = raw_impute(x_test_raw)

# Feature extraction
try:
    print("Loading existing features")
    extracted_features = pd.read_pickle("./x_train_extracted_features.pkl")
except FileNotFoundError:
    print("Feature extraction...")
    extracted_features = extract_features(x, column_id="pid", column_sort="Time", n_jobs=4)
    impute(extracted_features)
    extracted_features.to_pickle("./x_train_extracted_features.pkl")

# Feature selection
print("Feature selection...")
x_train_class = pd.DataFrame()
for label in labels_class:
    x_train_class[label] = select_features(extracted_features, y[label], ml_task='classification', n_jobs=4)

x_train_reg = pd.DataFrame()
for label in labels_class:
    x_train_reg[label] = select_features(extracted_features, y[label], ml_task='regression', n_jobs=4)

# Parameters Classification
param_grid = [
    {'randomforestclassifier__min_samples_split': [2, 4, 8],
     'randomforestclassifier__min_samples_leaf': [1, 2, 4]}
]

pipeline = Pipeline([('scale', StandardScaler),
                     ('classifier', RandomForestClassifier(class_weight='balanced'))])

# Initialize dataframe for results
results = pd.DataFrame()
results['pid'] = x_test['pid'].unique()

# Use Area Under the Receiver Operating Characteristic Curve (ROC AUC) as performance metric in Gridsearch
score = 'roc_auc'

for index, label in enumerate(labels_class):
    print("# Tuning hyper-parameters for label %s" % (label))

    clf = GridSearchCV(
        pipeline, param_grid, n_jobs=-1, scoring=score, cv=2
    )
    clf.fit(x_train_class[label], y.loc[:, label].values)

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
    #print("The scores are computed on the full evaluation set.")
    #print()
    #y_true, y_pred = y_test.loc[:, label], clf.predict(x_test)
    #print(classification_report(y_true, y_pred))

    results[label] = clf.predict_proba(x_test)


# Regression
score = 'r2'

#Pipeline Regression
estimator = make_pipeline(
        StandardScaler(),
        # RandomForestRegressor()
        linear_model.Lasso(max_iter=10000)
    )

# Parameters for Regression
param_grid_reg = [{'lasso__alpha': [0.1, 1, 10, 100]}]

for index, label in enumerate(labels_reg):
    print("# Tuning hyper-parameters for label %s" % (label))

    reg = GridSearchCV(
        estimator, param_grid_reg, scoring=score, n_jobs=-1
    )
    reg.fit(x_train_reg[label], y.loc[:, label])

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
    #print("The scores are computed on the full evaluation set.")
    #print()
    #y_true, y_pred = y_test.loc[:, label], reg.predict(x_test)
    #print(r2_score(y_true, y_pred))
    #print()
    results[label] = reg.predict(x_test)
print('Regression end')

# Save Results
results.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

print('Results saved as prediction.zip')