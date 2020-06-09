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

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score
from sklearn.utils import resample


def one_hot_encoding(df):
    acid_types = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U',
                  'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    df_one_hot = pd.DataFrame(columns=acid_types)
    print("One hot encoding ...")
    for index, row in df.iterrows():
        # Create the one hot encoded sample
        df_one_hot.loc[index] = [int(acid in row['Sequence']) for acid in acid_types]
    print("Done.")
    return df_one_hot


def write_csv(filename, data_):
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(data_)
    writeFile.close()
    print("[write_csv] Wrote data to: ", filename)
    return


def handle_imbalance(x, y, type_='downsample'):
    # Add labels to data
    data_ = x
    data_.insert(0, "Active", y)

    # Separate majority and minority classes
    df_majority = data_[data_['Active'] == 0]
    df_minority = data_[data_['Active'] == 1]

    if type_ is 'downsample':
        # Upsample minority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample with replacement
                                           n_samples=len(df_minority.index),  # to match majority class
                                           random_state=random_state)  # reproducible results
        # Combine majority class with upsampled minority class
        df_resampled = pd.concat([df_minority, df_majority_downsampled])
    if type_ is 'upsample':
        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=len(df_majority.index),  # to match majority class
                                         random_state=random_state)  # reproducible results
        # Combine majority class with upsampled minority class
        df_resampled = pd.concat([df_majority, df_minority_upsampled])

    # Separate x and y
    y_balanced = df_resampled['Active']
    x_balanced = df_resampled.drop('Active', axis=1)

    # Display new class counts
    print("Resampled with strategy: ", type_, "Value counts: ", y_balanced.value_counts())
    return x_balanced, y_balanced


if __name__ == '__main__':
    # Load data
    training_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    random_state = 3

    # One hot encoding
    # X_submission = one_hot_encoding(test_data)
    # x_training = one_hot_encoding(training_data)
    X_submission = pd.read_csv('X_test_one_hot.csv').iloc[:, 1:]  # iloc to get rid of the index column
    x_training = pd.read_csv('X_train_one_hot.csv').iloc[:, 1:]  # iloc to get rid of the index column
    y_training = training_data['Active']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        x_training, y_training, test_size=0.33, random_state=random_state, stratify=y_training)

    # Dealing with imbalance
    X_train, y_train = handle_imbalance(X_train, y_train, type_='downsample')

    # Instantiate models
    print("Fitting models")
    models = {
        "Multilayer Perceptron Classifier": MLPClassifier(random_state=random_state, max_iter=300, verbose=1,
                                                          hidden_layer_sizes=1000),
        # TODO try class weighting
        "Support Vector Classifier": SVC(verbose=1, random_state=random_state),
        "Linear Support Vector Classifier": LinearSVC(verbose=1, random_state=random_state)
    }

    # Train models
    for model_name, model in models.items():
        print("Training", model_name, "...")
        model.fit(X_train, y_train)

    # Evaluate
    for model_name, model in models.items():
        y_test_hat = model.predict(X_test)
        print("F1_score of", model_name, f1_score(y_true=y_test, y_pred=y_test_hat))

    # # Predict on the test set and save to .csv
    # y_submission = clf.predict(X_submission)
    # np.savetxt('Y_submission_vanilla_MLP_downsampled.csv', y_submission, delimiter=",", fmt='%d')
