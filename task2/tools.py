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
