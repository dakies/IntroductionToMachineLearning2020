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
import sklearn.preprocessing
from sklearn.covariance import EllipticEnvelope

import tools
import model_fitting
import inspection


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

    load = False
    if load:
        # Load impunated data
        patient_data_impunated = pd.read_csv('data/vital_signs_median_impunated_age100_removed_outliers_labeled.csv')
    else:
        print("\nImpunating data...")
        starting_time, cts = time.time(), 0
        for _, patient in patient_data:
            # Replace age with median if it is 100
            # if patient["Age"].mean() == 100.0:
            #     patient["Age"] = median_all_patients["Age"]

            # Fill all NaNs with patient median of variable
            patient = patient.fillna(patient.median(skipna=True))

            # Fill with hospital median if a variable is all NaNs
            if patient.isnull().sum().sum():
                patient = patient.fillna(median_all_patients)

            patient_data_impunated = patient_data_impunated.append(patient)
            # print progress
            tools.progress_bar(cts, len(patient_data), start_time=starting_time)
            cts = cts + 1

        # print("First patient after impunation: \n", patient_data_impunated.head(12))
        # Get rid of outliers
    #     print("Fitting ellipse")
    #     fit_elliptic_envelope = EllipticEnvelope(contamination=0.2)
    #     labels_outliers = fit_elliptic_envelope.fit_predict(patient_data_impunated)
    #     patient_data_impunated.insert(1, "outlier", labels_outliers, True)
    #
    # # Go through patients and get rid of outlier patients
    # patient_data_impunated_no_outliers = patient_data_impunated.head(0)
    # starting_time, cts = time.time(), 0
    # for _, patient in patient_data_impunated.groupby(by="pid"):
    #     # print progress
    #     tools.progress_bar(cts, len(patient_data_impunated), start_time=starting_time)
    #     cts = cts + 1
    #
    #     # Cancel outliers
    #     if any(patient["outlier"] == -1):
    #         continue
    #     else:
    #         patient_data_impunated_no_outliers = patient_data_impunated_no_outliers.append(patient)
    patient_data_impunated_no_outliers = patient_data_impunated

    one_to_twelve = np.tile(np.linspace(1, 12, 12, dtype=int), int(len(patient_data_impunated_no_outliers) / 12))
    patient_data_impunated_no_outliers.insert(1, "hour", one_to_twelve, True)
    vital_signs = patient_data_impunated_no_outliers.set_index(['pid', 'hour'])

    # Save vital signs to file
    if save:
        filename = 'data/vital_signs_median_impunated_outliers_and_age100_removed.csv'
        vital_signs.to_csv(filename)
        print('Saved file to ', filename)

    return vital_signs, patient_data_impunated


def variable_scaling(x, y):
    # scaler_x = sklearn.preprocessing.MinMaxScaler()
    # scaler_x = sklearn.preprocessing.StandardScaler()
    scaler_x = sklearn.preprocessing.RobustScaler()
    x.loc[:, :] = scaler_x.fit_transform(x.loc[:, :])
    # x = scaler_x.fit_transform(x)
    # print(scaler_x.data_min_)
    # print(scaler_x.data_max_)
    return x
