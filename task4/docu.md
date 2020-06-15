Setup:

First you have to install keras
for the download of the model to work I had to follow the first answer here: https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664
then I had to install pillow: https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664

Profiling shows that df creation and appending takes lots of time
(call)|         1|   0.00285006|   0.00285006| 18.00%|# /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/pandas/core/frame.py:860 iterrows
   115|         0|            0|            0|  0.00%|            # index = list_of_image_names.index(row['A']) # Find the index of an imagename in the feature list
   116|         0|            0|            0|  0.00%|            # list_of_features[index] # get the feature at that index
   117|         0|            0|            0|  0.00%|            features = np.concatenate((
   118|         0|            0|            0|  0.00%|                list_of_features[list_of_image_names.index(row['A'])].reshape(-1),
(call)|         1|  0.000431061|  0.000431061|  2.72%|# /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/pandas/core/series.py:1068 __getitem__
   119|         0|            0|            0|  0.00%|                list_of_features[list_of_image_names.index(row['B'])].reshape(-1),
(call)|         1|  0.000176907|  0.000176907|  1.12%|# /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/pandas/core/series.py:1068 __getitem__
   120|         0|            0|            0|  0.00%|                list_of_features[list_of_image_names.index(row['C'])].reshape(-1)
(call)|         1|  0.000185966|  0.000185966|  1.17%|# /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/pandas/core/series.py:1068 __getitem__
(call)|         1|  5.88894e-05|  5.88894e-05|  0.37%|# <__array_function__ internals>:2 concatenate
   121|         0|            0|            0|  0.00%|            ))
   122|         0|            0|            0|  0.00%|            train_features = train_features.append(pd.DataFrame(features.reshape(1, -1)))
(call)|         1|   0.00257301|   0.00257301| 16.25%|# /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/pandas/core/frame.py:397 __init__
(call)|         1|   0.00350189|   0.00350189| 22.11%|# /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/pandas/core/frame.py:6999 append
   123|

  TODO: Make sure to split train/test set such that there are no common images --> it needs to learn taste from image?
