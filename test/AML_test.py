import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import pandas as pd
import copy as cp
import shelve

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample



def read_csv(filename):
    with open((filename + '.csv'), 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        data_np = np.asarray(data[1:], dtype=float)
        readFile.close()
        return data_np

def write_csv(filename, data):
    with open((filename + '.csv'), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['Id', 'y'])
        writer.writerows(data)
    writeFile.close()
    print("Wrote data to: ", filename)


def panda_check():
    X_train_pd_eeg1 = pd.read_csv('train_eeg1.csv')
    X_train_pd_eeg2 = pd.read_csv('train_eeg2.csv')
    X_train_pd_emg = pd.read_csv('train_emg.csv')
    Y_train_pd = pd.read_csv('train_labels.csv')

    check_data(X_train_pd_eeg1)
    check_data(X_train_pd_eeg2)
    check_data(X_train_pd_emg)
    check_data(Y_train_pd)


def check_data(pandas_variable):
    ds = pandas_variable
    print(ds.head())
    print("NaN before cleaning: ", ds.isnull().sum().sum())


def preprocess(X_np, min_freq, max_freq):
    # Create power spectrum of the inputs
    # Optimally: check for visual correlation with the labels
    # Take content inside the first 1 -- 20-30 Hz

    fft = np.fft.fft(X_np, axis=1)
    ps = abs(fft)**2
    freqs = np.fft.fftfreq(512, 1/128)
    idx = np.argsort(freqs)

    ps_freq = ps[:, idx]
    ps_band = ps_freq[:, (256+min_freq):256+max_freq]
    ps_norm = np.linalg.norm(ps_band, axis=1, ord=1, keepdims=1)

    # print(ps_norm[1])
    # plt.plot(ps_freq[1, :])
    # plt.show()

    # return the normalized power spectrum for frequencies
    # from 1 to max_freq
    return ps_band/ps_norm

    # print(ps_norm[1])
    # plt.plot(ps_band[1, :])
    # plt.plot(ps_band_normal[1, :])
    # plt.show()
    # print(ps_freq.shape)


def fit_model_SVC(X, y):
    # split data
    # X_train, X_test, y_train, y_test = \
     #   train_test_split(X, y, random_state=0, test_size=0.3)

    clf = SVC(gamma='auto', class_weight='balanced')
    clf.fit(X, y)
    y_hat = clf.predict(X)

    print("Classification score on training split:",
          balanced_accuracy_score(y, y_hat))

    return clf


def fit_model(X, y):
    # split data
    #X_train, X_test, y_train, y_test = \
    #    train_test_split(X, y, random_state=0, test_size=0.3)

    clf = GradientBoostingClassifier()
    clf.fit(X, y)
    y_hat = clf.predict(X)

    print("Classification score on training split:",
          balanced_accuracy_score(y, y_hat))

    return clf


def load_test_data():
    # Load test data
    X_test_np_eeg1 = read_csv('test_eeg1')[:, 1:]
    X_test_np_eeg2 = read_csv('test_eeg2')[:, 1:]
    X_test_np_emg = read_csv('test_emg')[:, 1:]
    print("Test data loaded!")
    print(X_test_np_eeg1.shape, X_test_np_eeg2.shape,
          X_test_np_emg.shape)

    # Preprocessing
    X_test_eeg1 = preprocess(X_test_np_eeg1, 1, 25)
    X_test_eeg2 = preprocess(X_test_np_eeg2, 1, 25)
    X_test_eegs = np.hstack((
        X_test_eeg1, X_test_eeg2))
    X_test_emg = preprocess(X_test_np_emg, 2, 30)
    X_test = np.hstack((X_test_eegs, X_test_emg))
    print(X_test.shape)
    print("Training data preprocessed!")

    return X_test


if __name__ == '__main__':
    # Read csv's to panda and check values
    # panda_check()

    # Read csv's into np arrays
    # X_train_np_eeg1 = read_csv('train_eeg1')[:, 1:]
    # X_train_np_eeg2 = read_csv('train_eeg2')[:, 1:]
    # X_train_np_emg = read_csv('train_emg')[:, 1:]
    # Y_train_np = read_csv('train_labels')[:, 1:]
    # small
    X_train_np_eeg1 = read_csv('train_eeg1_small')[:, 1:]
    X_train_np_eeg2 = read_csv('train_eeg2_small')[:, 1:]
    X_train_np_emg = read_csv('train_emg_small')[:, 1:]
    Y_train_np = read_csv('train_labels')[:96, 1:]
    print("Training data loaded!")
    print(X_train_np_eeg1.shape, X_train_np_eeg2.shape,
          X_train_np_emg.shape, Y_train_np.shape)

    # # Balance training set with subsampling
    # shengo = Y_train_np.astype(int)
    # dist = np.bincount(shengo.ravel())
    # print('Distribution before subsampling:', dist)
    # # output: [ 0 63  5 28]
    # X_one = np.where(Y_train_np == 1)[0]
    # X_two = np.where(Y_train_np == 2)[0]
    # X_three = np.where(Y_train_np == 3)[0]
    # size = np.amin(dist[1:])
    # indexes = []
    # for i in range(size):
    #     indexes.append(X_one[int(float(i)/size*len(X_one))])
    #     indexes.append(X_two[int(float(i)/size*len(X_two))])
    #     indexes.append(X_three[int(float(i)/size*len(X_three))])
    # indexes_np = np.asarray(indexes)
    # Y_train = Y_train_np[indexes_np]
    # shengo2 = Y_train.astype(int)
    # print('Distribution after subsampling:',
    #       np.bincount(shengo2.ravel()))
    # print('Subsampling completed!')

    # Preprocessing
    X_train_eeg1 = preprocess(X_train_np_eeg1, 1, 25)
    X_train_eeg2 = preprocess(X_train_np_eeg2, 1, 25)
    X_train_eegs = np.hstack((
        X_train_eeg1, X_train_eeg2))
    X_train_emg = preprocess(X_train_np_emg, 2, 30)
    X_train = np.hstack((X_train_eegs, X_train_emg))
    print("Training data preprocessed!")
    # X_train_eeg1 = preprocess(X_train_np_eeg1[indexes_np, :], 1, 25)
    # X_train_eeg2 = preprocess(X_train_np_eeg2[indexes_np, :], 1, 25)
    # X_train_eegs = np.hstack((
    #      X_train_eeg1, X_train_eeg2))
    # X_train_emg = preprocess(X_train_np_emg[indexes_np, :], 2, 30)
    # X_train = np.hstack((X_train_eegs, X_train_emg))
    # print("Training data preprocessed!")

    # Fit model
    model = fit_model_SVC(X_train, Y_train_np)
    print("Model fit to training data!")

    # Evaluate
    X_test = load_test_data()
    Y_hat_test = model.predict(X_test)
    print(Y_hat_test)
    Y_hat_test = Y_hat_test.reshape(Y_hat_test.size, 1)
    print(Y_hat_test)

    # Write CSV
    index = np.arange(Y_hat_test.size).reshape(Y_hat_test.size, 1)
    print(index.shape, Y_hat_test.shape)
    write_csv('Y_test_SVC_all_data', np.hstack((index.astype(int), Y_hat_test.astype(int))))






