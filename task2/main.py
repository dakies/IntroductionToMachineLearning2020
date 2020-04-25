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
import model_fitting
import inspection
import preprocessing
import data_loading

""" ---- Medical Events prediction ----

    Data: 12 hours per patient, vital signs and test results
    Subtask 1: Predict whether medical tests will be ordered (classification with softmax) 
    Subtask 2: Predict whether sepsis will occur (classification with softmax)
    Subtask 3: predict future means of vital signs (regression?)
"""


if __name__ == '__main__':
    # Load raw feature data
    vital_signs_raw, tests = data_loading.load_features("data/train_features.csv")
    # Load preprocessed feature data
    vital_signs_preprocessed_pd = pd.read_csv("data/vital_signs_median_impunated.csv")
    vital_signs_preprocessed_pd = vital_signs_preprocessed_pd.set_index(['pid', 'hour'])
    x_vital_pd = vital_signs_preprocessed_pd.unstack(level=1)

    # Load label data
    y_vital_pd, y_tests_pd, y_sepsis_pd = data_loading.load_labels("data/train_labels.csv")
    y_vital_pd = y_vital_pd.set_index(['pid'])
    y_vital_pd = y_vital_pd.sort_index(axis=0, level=0)

    # Inspect data
    # inspection.inspect_vital_signs(vital_signs_raw)

    # Preprocess vital signs
    print("\n------ PREPROCESSING ------")
    # vital_signs_preprocessed_pd = preprocessing.preprocess_vital_signs(vital_signs_raw)
    # x_vital_pd = vital_signs_preprocessed_pd.unstack(level=1)
    # a = x_vital_per_patient.loc[:, 'Time':'Temp'].to_numpy()
    # Check that the indices match for x and y
    if all(x_vital_pd.index == y_vital_pd.index):
        print('Indices match')

    # Split into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_vital_pd, y_vital_pd, test_size=0.33, random_state=42)

    # Training
    print("\n------ TRAINING & VALIDATION ------")
    # Train linear regression model for every label
    model_fitting.linreg_ls_lasso_ridge(X_train, X_test, y_train, y_test)

    # # validate performance
    # print("\n------ VALIDATION ------")
    # print("Overall R2 score on test set:", reg.score(X_test[:, 1:], y_test[:, 1:]))
    # for column in range(np.shape(y_vital)[1] - 1):
    #     print('Individual score:', r2_score(y_hat[:, column], y_test[:, column + 1]))
    #
    # # TODO Compare to individual regressions
