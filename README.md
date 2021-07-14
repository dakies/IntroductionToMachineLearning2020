# IML_2020
Projects from the Introduction to Machine Learning course 2020
Grade 5.85 out of 6
## Project 1
Warm up
Simple least squares with feature engineering, regularisation ...
## Project 2
In this task, as an illustration of a real-world problem, you are asked to predict the evolution of hospital patients' states and needs during their stay in the Intensive Care Unit (ICU). For each patient, you are provided with the monitoring information collected during the first 12h of their stay. From the data, you first need to extract meaningful features describing this 12h stay. Then, for each sub-task, you should select an appropriate model to train on your pre-processed data. You will face the typical challenges of working with real medical data: missing features and imbalanced classification, predicting rarely-occuring events

Medical Event Prediction
Data: 12 hours per patient (time series), vital signs and test results
Subtask 1: Predict whether medical tests will be ordered 
Subtask 2: Predict whether sepsis will occur 
Subtask 3: predict future means of vital signs 

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
## Project 3
Classify mutations of a human antibody protein into active (1) and inactive (0) based on the provided mutation information. Under active mutations the protein retains its original function, and inactive mutation cause the protein to lose its function. The mutations differ from each other by 4 amino acids in 4 respective sites. The sites or locations of the mutations are fixed. The amino acids at the 4 mutation sites are given as 4-letter combinations, where each letter denotes the amino acid at the corresponding mutation site. Amino acids at other places are kept the same and are not provided.

For example, FCDI corresponds to amino acid F (Phenylanine) being in the first site, amino acid C (Cysteine) being in the second site and so on. The Figure 2 gives translation from symbols to amino acid chemical names for the interested students. The biological and chemical aspects can be abstracted to solve this task.
## Project 4
In this task, you will make decisions on food taste similarity based on images and human judgements.
We provide you with a dataset of images of 10.000 dishes, a sample of which is shown below.
Pipeline consiting of 
1)Feature extraction using deep learning approach, by removing the last fully connected layer.
2)Classfication using a variety of diffferent classifer
3) Use Bayesian optimisation to find optimal parameters for classifers
Goal: Create model to identify wich images of food taste the most similar
