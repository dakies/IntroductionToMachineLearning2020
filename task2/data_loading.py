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

    y_tests_ = data[['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]
    y_vital_signs = data[['pid', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
    y_sepsis = data[['pid','LABEL_Sepsis']]

    return y_vital_signs, y_tests_, y_sepsis
