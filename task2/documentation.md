# IML_2020 task 2
Introduction to Machine Learning course 2020

results task 2 here https://docs.google.com/spreadsheets/d/1LFfaXcjigT8bla9nSAW6k1N5Eh8rL38APm4yRKEtR_8/edit?usp=sharing

# New start with xgb
Starting XGB Classification...
ROC score for label LABEL_BaseExcess: 0.8478
ROC score for label LABEL_Fibrinogen: 0.6219
ROC score for label LABEL_AST: 0.6103
ROC score for label LABEL_Alkalinephos: 0.6342
ROC score for label LABEL_Bilirubin_total: 0.6156
ROC score for label LABEL_Lactate: 0.6649
ROC score for label LABEL_TroponinI: 0.6991
ROC score for label LABEL_SaO2: 0.7041
ROC score for label LABEL_Bilirubin_direct: 0.5539
ROC score for label LABEL_EtCO2: 0.7797
ROC score for label LABEL_Sepsis: 0.5026
Starting XGB Regression...
R2 score for label LABEL_RRate: 0.4019
R2 score for label LABEL_ABPm: 0.6191
R2 score for label LABEL_SpO2: 0.2481
R2 score for label LABEL_Heartrate: 0.6509

with impuation:
Starting XGB Classification...
ROC score for label LABEL_BaseExcess: 0.7665
ROC score for label LABEL_Fibrinogen: 0.6153
ROC score for label LABEL_AST: 0.6240
ROC score for label LABEL_Alkalinephos: 0.6038
ROC score for label LABEL_Bilirubin_total: 0.5994
ROC score for label LABEL_Lactate: 0.6631
ROC score for label LABEL_TroponinI: 0.6714
ROC score for label LABEL_SaO2: 0.6836
ROC score for label LABEL_Bilirubin_direct: 0.5358
ROC score for label LABEL_EtCO2: 0.7630
ROC score for label LABEL_Sepsis: 0.5066
Starting XGB Regression...
R2 score for label LABEL_RRate: 0.3535
R2 score for label LABEL_ABPm: 0.5916
R2 score for label LABEL_SpO2: 0.2625
R2 score for label LABEL_Heartrate: 0.6072

applying class weighting does not make it better
XGBRF also doesn't make it better

---- Medical Events prediction ----

Data: 12 hours per patient, vital signs and test results
Subtask 1: Predict whether medical tests will be ordered (classification with softmax)
Subtask 2: Predict whether sepsis will occur (classification with softmax)
Subtask 3: predict future means of vital signs (regression?)
        - Linear regression on data impunated with median - gives quite bad values, time does not have a big impact, all features train better when
        using all features except RRate --> Lasso and Ridge do not perform any better
            random seed  for train test split: 42
            Linear regression
            R2 score -0.453 for LABEL_RRate trained only with RRate
            R2 score -0.366 for LABEL_RRate trained with all features
            R2 score -42.913 for LABEL_ABPm trained only with ABPm
            R2 score 0.397 for LABEL_ABPm trained with all features
            R2 score -58.514 for LABEL_SpO2 trained only with SpO2
            R2 score -0.540 for LABEL_SpO2 trained with all features
            R2 score -37.208 for LABEL_Heartrate trained only with Heartrate
            R2 score 0.503 for LABEL_Heartrate trained with all features
            Lasso regression
            R2 score -0.863 for LABEL_RRate trained only with RRate
            R2 score -0.778 for LABEL_RRate trained with all features
            R2 score -109.241 for LABEL_ABPm trained only with ABPm
            R2 score 0.376 for LABEL_ABPm trained with all features
            R2 score -4383.238 for LABEL_SpO2 trained only with SpO2
            R2 score -2.635 for LABEL_SpO2 trained with all features
            R2 score -46.864 for LABEL_Heartrate trained only with Heartrate
            R2 score 0.492 for LABEL_Heartrate trained with all features
            Ridge regression
            R2 score -0.453 for LABEL_RRate trained only with RRate
            R2 score -0.366 for LABEL_RRate trained with all features
            R2 score -42.914 for LABEL_ABPm trained only with ABPm
            R2 score 0.397 for LABEL_ABPm trained with all features
            R2 score -58.514 for LABEL_SpO2 trained only with SpO2
            R2 score -0.540 for LABEL_SpO2 trained with all features
            R2 score -37.208 for LABEL_Heartrate trained only with Heartrate
            R2 score 0.503 for LABEL_Heartrate trained with all features
        - Impunation with 0 is even worse for all models and training settings
        - do a mean shift and scale for features: tried with 3 different rescalings, nothing improves
        - trying to predict with median instead of all 12 values
        - probably need to find some feature transformation / kernels

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


Here are some meta-data on the meaning and physical units of each variable in the data-set,
in the first paragraph the vital signs and EtCO2 is treated, in the second paragraph the
(laboratory) measurement tests.

'Temp' is the body temperature [Celsius]
'RRate' is the respiration rate of the patient [breath/min]
ABPm, ABPd, ABPs are the mean arterial, diastolic and systolic blood pressures of the patient [mmHg],
'Heartrate' is the number of heart beats per minute [heart beats/min],
'SpO2' is pulse oximetry-measured oxygen saturation of the blood [%].
'EtCo2' is the CO2 pressure during expiration [mmHg].

PTT: a test which measures the time it takes for a blood clot to form [sec.]
BUN: Blood urea nitrogen concentration [in mg per dl]
Lactate: Lactate acid concentration [in mg per dl]
Hgb: Haemoglobin concentration [g per dl]
HCO3: Bicarbonate concentration [mmol per l]
BaseExcess: Base excess measured in a blood gas analysis [mmol per l]
Fibrinogen: A protein produced by the liver. This protein helps stop bleeding by helping blood clots to form. Concentration [mg per dl]
Phosphate: Phosphate concetration [mg per dl]
WBC: White blood cell count in blood [number of 1000s per microliter]
Creatinine: Serum creatinine concentration used to determine renal function [mg per dl]
PaCO2: Partial pressure of CO2 in arterial blood [mmHg] indicates effectiveness of lung function
AST: Aspartate transaminase, a clinical test determining liver health [International unit per liter, biological activity]
FiO2: Fraction of inspired oxygen in %
Platelets: Thromocyte count in blood [numbers of 1000s per microliter]
SaO2: Oxygen saturation in arterial blood analyzed with blood gas analysis [%]
Glucose: Concentration of serum glucose [in mg per dl]
Magnesium: Concentration of magnesium in blood [mmol per dl]
Potassium: Concentration of potassium in blood [mmol per liter]
Calcium: Concentration of calcium in blood [mg per dl]
Alkalinephos: Biological activity of the enzyme Alkaline phosphotase [International unit per liter]
Bilirubin_direct: Bilirubin concentration of conjugated bilirubin [mg per dl]
Chloride: Chloride concentration in blood [mmol per l]
Hct: Volume percentage of red blood cells in the blood [%]
Bilirubin_total: Bilirubin concentration including conjugated / unconjugated bilirubin [mg per dl]
TroponinI: Concentration of troponin in the blood [ng per ml]
pH: Measurement of the acidity or alkalinity of the blood, with a standard unit for pH.

Each of these measurements has a standard measurement modality and you can assume it was used in the majority of the cases,
i.e. various measurements are taken from a blood sample which is then analyzed in the laboratory, for example.