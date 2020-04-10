import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import time
import pandas as pd
import copy as cp
import shelve

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge, BayesianRidge

import tools

""" ---- Medical Events prediction ----

    Data: 12 hours per patient, vital signs and test results
    Subtask 1: Predict whether medical tests will be ordered (classification with softmax) 
    Subtask 2: Predict whether sepsis will occur (classification with softmax)
    Subtask 3: predict future means of vital signs (regression?)
    lots of missing data (especially in the tests) 
    class occurrence imbalance
    predicting rare events
    
    watch toturials, understand AUC metric
    Deal with NaN's
    Make simple models for subtasks separately
    make a test set from available training data
    engineer features to encode the temporal information

"""


def algorithm():
    # Description
    desc = 'test1'

    # Read inputs
    data = pd.read_csv("data/train_features.csv")
    print(data.head())

    # # Write result
    # print(w.reshape(-1, 1))
    # write_csv(('task1b/data/result_' + desc + '_' + mode +'.csv'), w.reshape(-1, 1))
    # print('Algorithm succeeded.')


if __name__ == '__main__':
    # algorithm()

    # Read inputs
    data = pd.read_csv("data/train_features.csv")
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Split data into vital signs and tests
    vital_signs = data[['pid', 'Time', 'Age', 'Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'RRate', 'Temp']]
    tests = data[['pid', 'Time', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess', 'Fibrinogen',
                  'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
                  'Magnesium', 'Potassium', 'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
                  'Bilirubin_total', 'TroponinI', 'pH']]

    # Inspect NaN's
    print(vital_signs.isnull().sum() / len(vital_signs))

    # Inspect NaN's
    print(tests.isnull().sum() / len(tests))

    # pd.set_option('display.max_colwidth', -1)
    print(tests.head(100))
