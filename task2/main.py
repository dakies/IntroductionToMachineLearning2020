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
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.covariance import EllipticEnvelope


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

def pairgrid():
    # # inspection.inspect_w_seaborn(y_vital_pd)
    # inspection.inspect_vital_signs(vital_signs_raw)
    # # fig, axes = plt.subplots(4, 1)
    # linspacee = np.linspace(0, len(vital_signs_preprocessed_pd.index), 100, endpoint=False, dtype=int)
    # indexes = [vital_signs_preprocessed_pd.index.values[entry][0] for entry in linspacee]
    # # fig.set_size_inches(10, 5)
    #
    # labels_x = ['Age', 'Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'RRate', 'Temp']
    # labels_y = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    # median_hr = [vital_signs_preprocessed_pd.loc[entry[0]]['Heartrate'].median()
    #              for entry in vital_signs_preprocessed_pd.index.values]

    # x_vals = vital_signs_preprocessed_pd.groupby(by="pid").max() - vital_signs_preprocessed_pd.groupby(by="pid").min()
    x_vals = x_vital_pd_scaled.groupby(by="pid").mean()
    # x_vals.loc[:]["Age"] = vital_signs_preprocessed_pd.groupby(by="pid").mean().loc[:]["Age"]
    indices = np.linspace(1, len(x_vals.index), endpoint=False, num=10000, dtype=int)
    indices = x_vals.index[indices]

    x_vals = x_vals.loc[indices]
    y_vals = y_vital_pd.loc[indices]

    # Get rid of outliers
    # print("Fitting ellipse")
    # fit_elliptic_envelope = EllipticEnvelope()
    # labels_outliers = fit_elliptic_envelope.fit_predict(x_vals)
    # x_vals.insert(1, "outlier", labels_outliers, True)

    data = pd.concat([x_vals, y_vals], axis=1)

    x_labels = ['pid', 'Time', 'Age', 'Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'RRate', 'Temp']
    y_labels = ['pid', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    g = sns.PairGrid(data, x_vars=x_labels[1:], y_vars=y_labels[1:], hue="Age")
    g = g.map(plt.scatter, s=1)

    # sns.lmplot(x="Heartrate", y="LABEL_Heartrate", data=data)
    # sns.relplot(x="RRate", y="LABEL_RRate", data=data, hue="Age", legend='brief', palette="RdBu")
    # sns.relplot(x="SpO2", y="LABEL_SpO2", data=data, hue="Age", legend='brief', palette="RdBu")

    # sns.relplot(x=vital_signs_preprocessed_pd.groupby(by="pid").loc['Heartrate'].median(),
    #             y=y_vital_pd.loc[:]['LABEL_Heartrate'])

    # for pid in indexes:
    # for id, label in enumerate(labels):
    #     axes[0].plot(vital_signs_preprocessed_pd.loc[pid][label], vital_signs_preprocessed_pd.loc[pid]['Temp'])
    #
    #     plt.show()
    # plt.waitforbuttonpress(0)


if __name__ == '__main__':
    # Load raw feature data
    vital_signs_raw, tests = data_loading.load_features("data/train_features.csv")
    # Preprocess data
    # vital_signs_preprocessed_pd, patient_data_impunated = \
    #     preprocessing.preprocess_vital_signs(vital_signs_raw, save=True)
    # Load preprocessed feature data
    vital_signs_preprocessed_pd = pd.read_csv('data/vital_signs_median_impunated_outliers_and_age100_removed.csv')
    vital_signs_preprocessed_pd = vital_signs_preprocessed_pd.set_index(['pid', 'hour'])
    # x_vital_pd = vital_signs_preprocessed_pd.unstack(level=1)

    # NOT SUCCESSFUL - Use only median instead of 12 features:
    x_vital_pd_mean = vital_signs_preprocessed_pd.groupby(by="pid").mean()
    x_vital_pd_max = vital_signs_preprocessed_pd.groupby(by="pid").max()
    x_vital_pd_min = vital_signs_preprocessed_pd.groupby(by="pid").min()
    x_vital_pd_var = vital_signs_preprocessed_pd.groupby(by="pid").var()
    x_vital_pd = pd.concat([x_vital_pd_mean.loc[:, 'Age':], x_vital_pd_var.loc[:, 'Age':],
                            x_vital_pd_max.loc[:, 'Age':], x_vital_pd_min.loc[:, 'Age':]], axis=1)

    # Load label data
    y_vital_pd, y_tests_pd, y_sepsis_pd = data_loading.load_labels("data/train_labels.csv")
    y_vital_pd = y_vital_pd.set_index(['pid'])
    y_vital_pd = y_vital_pd.sort_index(axis=0, level=0)
    y_tests_pd = y_tests_pd.set_index(['pid'])
    y_tests_pd = y_tests_pd.sort_index(axis=0, level=0)
    y_sepsis_pd = y_sepsis_pd.set_index(['pid'])
    y_sepsis_pd = y_sepsis_pd.sort_index(axis=0, level=0)

    # Align labels with the data
    y_vital_pd = y_vital_pd.loc[x_vital_pd.index]
    y_sepsis_pd = y_sepsis_pd.loc[x_vital_pd.index]
    y_tests_pd = y_tests_pd.loc[x_vital_pd.index]

    # Add classification labels to regression features
    x_vital_pd = pd.concat([x_vital_pd, y_sepsis_pd, y_tests_pd], axis=1)

    # Preprocess vital signs
    print("\n------ PREPROCESSING ------")
    # vital_signs_preprocessed_pd = preprocessing.preprocess_vital_signs(vital_signs_raw)
    # x_vital_pd = vital_signs_preprocessed_pd.unstack(level=1)

    # # NOT SUCCESSFUL - Scale features
    # x_vital_pd = preprocessing.\
    #     variable_scaling(x_vital_pd, y_vital_pd)
    # # x_vital_pd = x_vital_pd_scaled.unstack(level=1)

    # Inspect data
    # pairgrid()

    # Split into test and training sets
    # Check that the indices match for x and y
    if not all(x_vital_pd.index == y_vital_pd.index):
        print('Indices dont match')
        os.abort()
    X_train, X_test, y_train, y_test = train_test_split(
        x_vital_pd, y_vital_pd, test_size=0.33, random_state=42)

    # Training
    print("\n------ TRAINING & VALIDATION ------")
    # Train linear regression model for every label
    model_fitting.linreg_ls_lasso_ridge(X_train, X_test, y_train, y_test)

