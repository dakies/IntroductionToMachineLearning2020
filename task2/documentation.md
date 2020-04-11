# IML_2020
Introduction to Machine Learning course 2020

---- Medical Events prediction ----

Data: 12 hours per patient, vital signs and test results
Subtask 1: Predict whether medical tests will be ordered (classification with softmax)
Subtask 2: Predict whether sepsis will occur (classification with softmax)
Subtask 3: predict future means of vital signs (regression?)
lots of missing data (especially in the tests)
class occurrence imbalance
predicting rare events

check - [watch tutorials, understand AUC metric]
learn to deal with NaN's - tutorial 08.04
    if there is no information set zero, otherwise previous value
    Mechanisms of missingness (MCAR, MAR, NMAR) - use domain knowledge to determine the mechanism
    Dealing with it:
        regression wih existing values and adding some noise
        find similar cases do some nearest neighbor clustering
        if very sparse information, might want to think of summarizing it and not keeping raw data
        use CV to check things
learn about multiclass classification
handle class imbalance - toturial 25.03

extensions:
engineer features to encode the temporal information, get domain knowledge - tutorial 18.03, kernels
use classifier output for regression
feature scaling (use training distributions for mean and std.dev for testsets, PCA
    see notes from tutorial 04.03

"""