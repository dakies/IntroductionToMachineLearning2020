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


def read_csv(filename):
    panda_check(filename)
    i = 1
    print('Reading everything after row ' + str(i) + '.')
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        data_np = np.asarray(data[i:], dtype=float)
        readFile.close()
        return data_np


def write_csv(filename, data):
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(data)
    writeFile.close()
    print("Wrote data to: ", filename)


def panda_check(filename):
    data = pd.read_csv(filename)
    print(data.head())
    print("NaN before cleaning: ", data.isnull().sum().sum())


def algorithm():
    # Description
    desc = 'Vanillav1'

    # Read inputs
    data = read_csv('task1a/data/train.csv')
    y_train = data[:, 1]
    x_train = data[:, 2:]
    lambdas = [0.01, 0.1, 1, 10, 100]
    result = np.zeros((5, 1))

    # Iterate lambdas
    for i in range(5):
        # Declare estimator
        model = Ridge(lambdas[i])
        # Declare cross validation
        cv_results = cross_validate(model, x_train, y_train,
                                scoring='neg_mean_squared_error', cv=10)
        # Calculate mean RMSE
        cv_rmse = np.sqrt(abs(cv_results['test_score']))
        mean_rmse = np.mean(cv_rmse)

        result[i, 0] = mean_rmse

    # Write result
    print(result)
    write_csv(('task1a/data/result_' + desc + '.csv'), result)
    print('Algorithm succeeded.')


if __name__ == '__main__':
    algorithm()