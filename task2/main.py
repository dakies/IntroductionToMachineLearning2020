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


def preprocess_vital_signs(vital_signs):
    # Filling missing data with patient median or median of all patients
    # vital_signs = vital_signs.head(n=1200)  # Split into a smaller set

    # Group by patient
    patient_data = vital_signs.groupby(by="pid")
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

        # Write all features into a 1d array
        # TODO Now taking time as well
        patient_feature = patient.loc[:,
                          ['Age', 'Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'RRate',
                           'Temp']].to_numpy().ravel()
        pid = patient['pid'].mean()
        patient_feature = np.append(pid, patient_feature)
        patient_features_list.append(patient_feature)

        # print progress
        tools.progress_bar(cts, len(patient_data), start_time=starting_time)
        cts = cts + 1

    print("First patient after impunation: \n", patient_data_impunated.head(12))

    # Matrix with all features
    patient_features = np.asarray(patient_features_list)
    return patient_features


if __name__ == '__main__':
    # Load data
    vital_signs_raw, tests = load_features("data/train_features.csv")
    y_vital_pd, y_tests_pd, y_sepsis_pd = load_labels("data/train_labels.csv")
    y_vital = y_vital_pd.to_numpy()

    # Inspect data
    # inspect_vital_signs(vital_signs)

    # Preprocess vital signs
    print("\n------ PREPROCESSING ------")
    patient_features_vital = preprocess_vital_signs(vital_signs_raw)

    # Sort X and y by pid
    ind = np.argsort(patient_features_vital[:, 0])
    patient_features_vital = patient_features_vital[ind]
    ind2 = np.argsort(y_vital[:, 0])
    y_vital = y_vital[ind2]
    # Check pid's match in rows
    for index in range(np.shape(patient_features_vital)[0]):
        if not (y_vital[index, 0] == patient_features_vital[index, 0]):
            print('[Error] pids arent the same!')
            print(y_vital[index, 0], patient_features_vital[index, 0])

    # Split into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(
        patient_features_vital, y_vital, test_size=0.33, random_state=42)

    # Train linear regression model on all vital signs
    print("\n------ TRAINING ------")
    reg = LinearRegression().fit(X_train[:, 1:], y_train[:, 1:])
    y_hat = reg.predict(X_test[:, 1:])

    # validate performance
    print("\n------ VALIDATION ------")
    print("Overall R2 score on test set:", reg.score(X_test[:, 1:], y_test[:, 1:]))
    for column in range(np.shape(y_vital)[1] - 1):
        print('Individual score:', r2_score(y_hat[:, column], y_test[:, column + 1]))

    # TODO Compare to individual regressions
