import numpy as np
import csv
import pandas as pd


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
    desc = 'Vanillav1_val'

    # Read inputs
    data = read_csv('data/train.csv')
    y_train = data[:, 1]
    x_train = data[:, 2:]
    train_features = np.zeros((np.shape(x_train)[0], 21))

    # linear features
    train_features[:, 0:5] = x_train
    # quadratic features
    train_features[:, 5:10] = np.power(x_train, np.ones((np.shape(x_train)[0], 5))*2)
    # exponential features
    train_features[:, 10:15] = np.exp(x_train)
    # cosine features
    train_features[:, 15:20] = np.cos(x_train)
    # constant feature
    train_features[:, 20] = 1

    mode = 'ridge'
    if mode == 'LSQ':
        # Calculate weights with LSQ regression
        feat2inv = np.linalg.inv(np.matmul(train_features.T, train_features))
        w = np.matmul(feat2inv, (np.matmul(train_features.T, y_train)))
    elif mode == 'ridge':
        # Calculate weights with ridge regression
        lmda = 100
        mode = mode+str(lmda)
        feat2inv = np.linalg.inv(np.matmul(train_features.T, train_features)+np.eye(21)*lmda)
        w = np.matmul(feat2inv, (np.matmul(train_features.T, y_train)))
    else:
        print('ERROR: No regression algorithm specified!')
        return

    # Write result
    print(w.reshape(-1, 1))
    write_csv(('data/result_' + desc + '_' + mode +'.csv'), w.reshape(-1, 1))
    print('Algorithm succeeded.')


if __name__ == '__main__':
    algorithm()