# IML_2020
Introduction to Machine Learning course 2020

---- Medical Events prediction ----

Data: 12 hours per patient, vital signs and test results
Subtask 1: Predict whether medical tests will be ordered (classification with softmax)
Subtask 2: Predict whether sepsis will occur (classification with softmax)
Subtask 3: predict future means of vital signs (regression?)
        - Linear regression on data impunated with median - gives quite bad values, time does not have a big impact, all features train better when
        using all features except RRate --> Lasso and Ridge do not perform any better
            Linear regression
            R2 score -0.438 for LABEL_RRate trained only with RRate
            R2 score -0.347 for LABEL_RRate trained with all features
            R2 score -45.131 for LABEL_ABPm trained only with ABPm
            R2 score 0.388 for LABEL_ABPm trained with all features
            R2 score -50.294 for LABEL_SpO2 trained only with SpO2
            R2 score -0.314 for LABEL_SpO2 trained with all features
            R2 score -40.731 for LABEL_Heartrate trained only with Heartrate
            R2 score 0.508 for LABEL_Heartrate trained with all features
            Lasso regression
            R2 score -0.850 for LABEL_RRate trained only with RRate
            R2 score -0.775 for LABEL_RRate trained with all features
            R2 score -114.072 for LABEL_ABPm trained only with ABPm
            R2 score 0.367 for LABEL_ABPm trained with all features
            R2 score -2664.816 for LABEL_SpO2 trained only with SpO2
            R2 score -1.743 for LABEL_SpO2 trained with all features
            R2 score -53.048 for LABEL_Heartrate trained only with Heartrate
            R2 score 0.500 for LABEL_Heartrate trained with all features
            Ridge regression
            R2 score -0.438 for LABEL_RRate trained only with RRate
            R2 score -0.347 for LABEL_RRate trained with all features
            R2 score -45.131 for LABEL_ABPm trained only with ABPm
            R2 score 0.388 for LABEL_ABPm trained with all features
            R2 score -50.295 for LABEL_SpO2 trained only with SpO2
            R2 score -0.314 for LABEL_SpO2 trained with all features
            R2 score -40.731 for LABEL_Heartrate trained only with Heartrate
            R2 score 0.508 for LABEL_Heartrate trained with all features
        - Impunation with 0 is even worse for all models and training settings
        - probably need to find some feature transformation / kernels
        then do a mean shift and scale for features
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