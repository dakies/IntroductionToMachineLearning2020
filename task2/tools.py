import numpy as np
import csv
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt


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

# # Display an image
# def disp_img(img):
#     try:
#         if (img.data):
#             cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
#             cv.imshow('image', img)
#             cv.waitKey(0)
#             cv.destroyAllWindows()
#             return
#         else:
#             print('DISP_IMG: Image file contains no data.')
#             return
#     except TypeError:
#         print('DISP_IMG: Invalid image file provided.')
#         return

# Progress bar to visualize all sorts of processes
def progress_bar(counter, total_number_of_counts, start_time=None):
    # Calculates the progress in percent by dividing counter by total_number_of_counts
    # Then it adds a graphical representation of the percentage
    percent = (("\r%d" % ((counter + 1) * 100 / total_number_of_counts) + "% | ")
               + (int((counter + 1) * 10 / total_number_of_counts) * '=')
               + (int((10 - (counter + 1) * 10 / total_number_of_counts)) * ' ') + " | ")

    # If a starting time is provided then the progress bar will also output the time elapsed
    # and an extrapolation of the remaining time
    time_display = ''
    if start_time is not None:
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time * total_number_of_counts / (counter + 1) - elapsed_time
        time_display = (("%d" % (np.floor(elapsed_time/60))).zfill(2) + ":" + ("%d" % (elapsed_time%60)).zfill(2)
                 + " | -" + ("%d" % (np.floor(remaining_time / 60))).zfill(2)
                 + ":" + ('%d' % (remaining_time % 60)).zfill(2) + "  ")
    print((percent + time_display), end=" ", flush=True)


def read_csv(filename):
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        readFile.close()
    print('[read_csv] Reading ', filename, ' done.')
    return data


def write_csv(filename, data):
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(data)
    writeFile.close()
    print("[write_csv] Wrote data to: ", filename)
    return

