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
from sklearn.metrics import r2_score

import tools

""" ---- Medical Events prediction ----

    Data: 12 hours per patient, vital signs and test results
    Subtask 1: Predict whether medical tests will be ordered (classification with softmax) 
    Subtask 2: Predict whether sepsis will occur (classification with softmax)
    Subtask 3: predict future means of vital signs (regression?)
"""


def inspect_vital_signs(vital_signs):
    # Set output options
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Inspect NaN's
    print('Overall NaNs in vital signs [%] \n', vital_signs.isnull().sum() / len(vital_signs))
    print('looks like a 15 to 30 % NaNs. Are there patients with no data at all?')
    patient_data = vital_signs.groupby(by="pid")
    unique_vals = patient_data.count()  # Count unique values in object
    patients_w_no_data = unique_vals.transform(lambda x: (x == 0)).sum()  # count patients with 0 datapoints
    print('\n Patients w/o data in vital signs [%]: \n \n', patients_w_no_data / len(patient_data))
    print('Only very few patients with no data. Except for ABPd. These might be interesting patients though!')
    # unique_vals['ABPd'].plot.hist()
    print('For most of them we have all values but 20% have none.')
    vital_signs['ABPd'].plot.hist(bins=100)

    print(patient_data.mean())  #
    # print('NaNs in tests [%] \n', tests.isnull().sum() / len(tests))


def load_features(filename):
    # Read inputs
    data = pd.read_csv(filename)

    # Split data into vital signs and tests
    vital_signs_ = data[['pid', 'Time', 'Age', 'Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'RRate', 'Temp']]
    tests_ = data[['pid', 'Time', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess', 'Fibrinogen',
                   'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
                   'Magnesium', 'Potassium', 'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
                   'Bilirubin_total', 'TroponinI', 'pH']]

    return vital_signs_, tests_


def load_labels(filename):
    # Read inputs
    data = pd.read_csv(filename)

    y_tests_ = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    y_vital_signs = data[['pid', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
    y_sepsis = data[['LABEL_Sepsis']]

    return y_vital_signs, y_tests_, y_sepsis


def preprocess_vital_signs(vital_signs_raw_, save=None):
    # Filling missing data with patient median or median of all patients
    # vital_signs_raw_ = vital_signs_raw_.head(n=1200)  # Split into a smaller set

    # Group by patient
    patient_data = vital_signs_raw_.groupby(by="pid")
    median_all_patients = patient_data.median().median()

    # Fill nans with median of patient or all patients
    print("First patient before impunation: \n", patient_data.get_group(1))
    patient_data_impunated = patient_data.head(0)
    patient_features_list = []

    print("\nImpunating data...")
    starting_time, cts = time.time(), 0
    for _, patient in patient_data:
        # Fill all NaNs with patient median of variable
        patient = patient.fillna(patient.median(skipna=True))
        # Fill with hospital median if a variable is all NaNs
        if patient.isnull().sum().sum():
            patient = patient.fillna(median_all_patients)
        patient_data_impunated = patient_data_impunated.append(patient)

        # print progress
        tools.progress_bar(cts, len(patient_data), start_time=starting_time)
        cts = cts + 1

    print("First patient after impunation: \n", patient_data_impunated.head(12))

    one_to_twelve = np.tile(np.linspace(1, 12, 12, dtype=int), int(len(vital_signs_raw_) / 12))
    patient_data_impunated.insert(1, "hour", one_to_twelve, True)
    vital_signs = patient_data_impunated.set_index(['pid', 'hour'])

    # Save vital signs to file
    if save:
        vital_signs.to_csv('data/vital_signs_median_impunated.csv')
        print('Saved file to data/vital_signs_median_impunated.csv.')

    return vital_signs


if __name__ == '__main__':
    # Load raw feature data
    vital_signs_raw, tests = load_features("data/train_features.csv")
    # Load preprocessed feature data
    vital_signs_preprocessed_pd = pd.read_csv("data/vital_signs_median_impunated.csv")
    vital_signs_preprocessed_pd = vital_signs_preprocessed_pd.set_index(['pid', 'hour'])
    x_vital_pd = vital_signs_preprocessed_pd.unstack(level=1)

    # Load label data
    y_vital_pd, y_tests_pd, y_sepsis_pd = load_labels("data/train_labels.csv")
    y_vital_pd = y_vital_pd.set_index(['pid'])
    y_vital_pd = y_vital_pd.sort_index(axis=0, level=0)

    # Inspect data
    # inspect_vital_signs(vital_signs)

    # Preprocess vital signs
    print("\n------ PREPROCESSING ------")
    # vital_signs_preprocessed_pd = preprocess_vital_signs(vital_signs_raw, save=True)
    # a = x_vital_per_patient.loc[:, 'Time':'Temp'].to_numpy()
    # Check that the indices match for x and y
    if all(x_vital_pd.index == y_vital_pd.index):
        print('Indices match')

    # Split into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_vital_pd, y_vital_pd, test_size=0.33, random_state=42)

    # Training
    print("\n------ TRAINING ------")
    # Train linear regression model for every label
    labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    for index, label in enumerate(labels):
        if label is 'LABEL_RRate':
            # TODO a simple linear regression on RRate performs very badly, check data
            #   it actually looks hard to predict based only on past values.
            reg = LinearRegression().fit(X_train.loc[:, 'RRate'], y_train.loc[:, label])
            y_hat = reg.predict(X_test.loc[:, 'RRate'])
            print('Linear Regression - R2 score for', label, r2_score(y_hat, y_test.loc[:, label]))

        reg = LinearRegression().fit(X_train, y_train.loc[:, label])
        y_hat = reg.predict(X_test)
        print('Linear Regression - R2 score for', label, r2_score(y_hat, y_test.loc[:, label]))

    # # validate performance
    # print("\n------ VALIDATION ------")
    # print("Overall R2 score on test set:", reg.score(X_test[:, 1:], y_test[:, 1:]))
    # for column in range(np.shape(y_vital)[1] - 1):
    #     print('Individual score:', r2_score(y_hat[:, column], y_test[:, column + 1]))
    #
    # # TODO Compare to individual regressions
