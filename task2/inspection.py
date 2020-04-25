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

import tools
import model_fitting

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

def inspect_w_seaborn(df):
    # Violin plot
    sns.violinplot(data=df)