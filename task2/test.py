import numpy as np
import pandas as pd

# a = (np.random.random((5, 5))*10)
# print(a)
# ind = np.argsort(a[:, 0])
# print('\n', ind)
#
# a = a[ind,:]
# print('\n', a)

def load_features():
    # Read inputs
    data = pd.read_csv("data/train_features.csv")

    # Split data into vital signs and tests
    vital_signs_ = data[['pid', 'Time', 'Age', 'Heartrate', 'SpO2', 'ABPs', 'ABPm', 'ABPd', 'RRate', 'Temp']]
    tests_ = data[['pid', 'Time', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess', 'Fibrinogen',
                   'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
                   'Magnesium', 'Potassium', 'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
                   'Bilirubin_total', 'TroponinI', 'pH']]

    return vital_signs_, tests_


if __name__ == '__main__':
    vital_signs_raw, _ = load_features()
    indexes = np.tile(np.linspace(1, 12, 12, dtype=int), int(len(vital_signs_raw) / 12))
    vital_signs_raw.insert(0, "indexes", indexes, True)
    vital_signs = vital_signs_raw.set_index(['pid', 'indexes'])
    vital_signs = vital_signs.unstack(level=1)

    v = vital_signs.head(24)
    v_r = vital_signs_raw.head(24)