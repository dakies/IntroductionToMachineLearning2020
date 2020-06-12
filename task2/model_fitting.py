import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import time
import pandas as pd
import copy as cp
import shelve
import os
import time

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

import tools


def linreg_ls_lasso_ridge(X_train, X_test, y_train, y_test):

    labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    features = ['RRate', 'ABPm', 'SpO2', 'Heartrate']

    # Linear regression
    print('\nLinear regression')
    for index, label in enumerate(labels):
        # # Using single feature
        # reg = LinearRegression().fit(X_train.loc[:, features[index]], y_train.loc[:, label])
        # y_hat = reg.predict(X_test.loc[:, features[index]])
        # print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
        #       'for', label, 'trained only with', features[index])

        # Using all features
        reg = LinearRegression().fit(X_train, y_train.loc[:, label])
        y_hat = reg.predict(X_test)
        print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
              'for', label, 'trained with all features')

        # # Using all features except time
        # reg = LinearRegression().fit(X_train.loc[:, 'Age':'Temp'], y_train.loc[:, label])
        # y_hat = reg.predict(X_test.loc[:, 'Age':'Temp'])
        # print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
        #       'for', label, 'trained with all features except time')

    # Lasso regression
    print('\nLasso regression')
    alpha_lasso = 1
    for index, label in enumerate(labels):
        # # Using single feature
        # reg = Lasso(alpha=alpha_lasso).fit(X_train.loc[:, features[index]], y_train.loc[:, label])
        # y_hat = reg.predict(X_test.loc[:, features[index]])
        # print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
        #       'for', label, 'trained only with', features[index])

        # Using all features
        reg = Lasso(alpha=alpha_lasso).fit(X_train, y_train.loc[:, label])
        y_hat = reg.predict(X_test)
        print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
              'for', label, 'trained with all features')
    #
    #     # # Using all features except time
    #     # reg = Lasso(alpha=alpha_lasso).fit(X_train.loc[:, 'Age':'Temp'], y_train.loc[:, label])
    #     # y_hat = reg.predict(X_test.loc[:, 'Age':'Temp'])
    #     # print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
    #     #       'for', label, 'trained with all features except time')
    #
    # Ridge regression
    print('\nRidge regression')
    alpha_ridge = 1
    for index, label in enumerate(labels):
        # # Using single feature
        # reg = Ridge(alpha=alpha_ridge).fit(X_train.loc[:, features[index]], y_train.loc[:, label])
        # y_hat = reg.predict(X_test.loc[:, features[index]])
        # print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
        #       'for', label, 'trained only with', features[index])

        # Using all features
        reg = Ridge(alpha=alpha_ridge).fit(X_train, y_train.loc[:, label])
        y_hat = reg.predict(X_test)
        print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
              'for', label, 'trained with all features')
    #
    #     # # Using all features except time
    #     # reg = Ridge(alpha=alpha_ridge).fit(X_train.loc[:, 'Age':'Temp'], y_train.loc[:, label])
    #     # y_hat = reg.predict(X_test.loc[:, 'Age':'Temp'])
    #     # print('R2 score', "%.3f" % r2_score(y_hat, y_test.loc[:, label]),
    #     #       'for', label, 'trained with all features except time')