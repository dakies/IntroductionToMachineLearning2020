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

TRAINING WITH UNIQUE DATASETS

Datapoints lost during distinct test-train splitting 28563
Validation set size (483, 3)
Training set size (30469, 3)
Starting training of the model ...
Epoch 1/2
120/120 [==============================] - 167s 1s/step - loss: 0.9544 - accuracy: 0.6144 - val_loss: 0.5654 - val_accuracy: 0.7060
Epoch 2/2
120/120 [==============================] - 175s 1s/step - loss: 0.5191 - accuracy: 0.7422 - val_loss: 0.5967 - val_accuracy: 0.6915
x_train = np.load("NasNetLarge_features_balanced/x_train_NasNetLarge_avg_pool_balanced_all_features.npy")
# Split training data into a validation and test set that do not contain common images
train_set, val_set = train_val_split(train_triplets_balanced_with_y.iloc[:, 1:], threshold=500)
Datapoints lost during distinct test-train splitting 16075
Validation set size (45, 3)
Training set size (43395, 3)
model.evaluate(x_train[val_set.index, :], y_train[val_set.index])
2/2 [==============================] - 0s 25ms/step - loss: 0.4678 - accuracy: 0.7556
Out[5]: [0.4678163230419159, 0.7555555701255798]
Datapoints lost during distinct test-train splitting 19357
Validation set size (96, 3)
Training set size (40062, 3)
model.evaluate(x_train[val_set.index, :], y_train[val_set.index])
3/3 [==============================] - 0s 50ms/step - loss: 0.5574 - accuracy: 0.7083
Out[9]: [0.5573952794075012, 0.7083333134651184]

NasNetLarge_small_net_split_620_batch_1024

Datapoints lost during distinct test-train splitting 19357
Validation set size (96, 3)
Training set size (40062, 3)
Starting training of the model ...
Epoch 1/4
40/40 [==============================] - ETA: 0s - loss: 1.5046 - accuracy: 0.5110
Epoch 00001: val_accuracy improved from -inf to 0.58333, saving model to NasNetLarge_features_balanced/tmp_model
2020-06-16 17:03:16.938243: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
40/40 [==============================] - 171s 4s/step - loss: 1.5046 - accuracy: 0.5110 - val_loss: 0.6802 - val_accuracy: 0.5833
Epoch 2/4
40/40 [==============================] - ETA: 0s - loss: 0.6128 - accuracy: 0.6608
Epoch 00002: val_accuracy improved from 0.58333 to 0.66667, saving model to NasNetLarge_features_balanced/tmp_model
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
40/40 [==============================] - 156s 4s/step - loss: 0.6128 - accuracy: 0.6608 - val_loss: 0.5980 - val_accuracy: 0.6667
Epoch 3/4
40/40 [==============================] - ETA: 0s - loss: 0.5361 - accuracy: 0.7287
Epoch 00003: val_accuracy improved from 0.66667 to 0.69792, saving model to NasNetLarge_features_balanced/tmp_model
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
40/40 [==============================] - 167s 4s/step - loss: 0.5361 - accuracy: 0.7287 - val_loss: 0.5532 - val_accuracy: 0.6979
Epoch 4/4
40/40 [==============================] - ETA: 0s - loss: 0.4971 - accuracy: 0.7581
Epoch 00004: val_accuracy did not improve from 0.69792
40/40 [==============================] - 152s 4s/step - loss: 0.4971 - accuracy: 0.7581 - val_loss: 0.5546 - val_accuracy: 0.6875

NasNetLarge_small_net_split_620_batch_2048

Datapoints lost during distinct test-train splitting 19357
Validation set size (96, 3)
Training set size (40062, 3)
Starting training of the model ...
Epoch 1/8
20/20 [==============================] - ETA: 0s - loss: 3.3088 - accuracy: 0.5004
Epoch 00001: val_accuracy improved from -inf to 0.57292, saving model to NasNetLarge_features_balanced/tmp_model
2020-06-16 17:29:09.915623: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
20/20 [==============================] - 157s 8s/step - loss: 3.3088 - accuracy: 0.5004 - val_loss: 0.6887 - val_accuracy: 0.5729
Epoch 2/8
20/20 [==============================] - ETA: 0s - loss: 0.6807 - accuracy: 0.5597
Epoch 00002: val_accuracy improved from 0.57292 to 0.66667, saving model to NasNetLarge_features_balanced/tmp_model
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
20/20 [==============================] - 167s 8s/step - loss: 0.6807 - accuracy: 0.5597 - val_loss: 0.6431 - val_accuracy: 0.6667
Epoch 3/8
20/20 [==============================] - ETA: 0s - loss: 0.6165 - accuracy: 0.6664
Epoch 00003: val_accuracy did not improve from 0.66667
20/20 [==============================] - 156s 8s/step - loss: 0.6165 - accuracy: 0.6664 - val_loss: 0.5818 - val_accuracy: 0.6042
Epoch 4/8
20/20 [==============================] - ETA: 0s - loss: 0.5623 - accuracy: 0.7102
Epoch 00004: val_accuracy improved from 0.66667 to 0.72917, saving model to NasNetLarge_features_balanced/tmp_model
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
20/20 [==============================] - 156s 8s/step - loss: 0.5623 - accuracy: 0.7102 - val_loss: 0.5630 - val_accuracy: 0.7292
Epoch 5/8
20/20 [==============================] - ETA: 0s - loss: 0.5287 - accuracy: 0.7348
Epoch 00005: val_accuracy did not improve from 0.72917
20/20 [==============================] - 142s 7s/step - loss: 0.5287 - accuracy: 0.7348 - val_loss: 0.5653 - val_accuracy: 0.7083
Epoch 6/8
20/20 [==============================] - ETA: 0s - loss: 0.4974 - accuracy: 0.7582
Epoch 00006: val_accuracy did not improve from 0.72917
20/20 [==============================] - 143s 7s/step - loss: 0.4974 - accuracy: 0.7582 - val_loss: 0.5440 - val_accuracy: 0.7188
Epoch 7/8
20/20 [==============================] - ETA: 0s - loss: 0.4752 - accuracy: 0.7736
Epoch 00007: val_accuracy did not improve from 0.72917
20/20 [==============================] - 144s 7s/step - loss: 0.4752 - accuracy: 0.7736 - val_loss: 0.5408 - val_accuracy: 0.7188
Epoch 8/8
20/20 [==============================] - ETA: 0s - loss: 0.4590 - accuracy: 0.7842
Epoch 00008: val_accuracy did not improve from 0.72917
20/20 [==============================] - 144s 7s/step - loss: 0.4590 - accuracy: 0.7842 - val_loss: 0.6115 - val_accuracy: 0.7188

NasNetLarge_small_net_split_620_batch_1024_val

Datapoints lost during distinct test-train splitting 19357
Validation set size (96, 3)
Training set size (40062, 3)
Starting training of the model ...
Epoch 1/4
40/40 [==============================] - ETA: 0s - loss: 1.9032 - accuracy: 0.5117
Epoch 00001: val_accuracy improved from -inf to 0.54167, saving model to NasNetLarge_features_balanced/tmp_model
2020-06-16 18:00:39.364492: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /Users/leivandresen/Documents/ETH/Master/3_Semester/AML/Projects/code/venv/lib/python3.7/site-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
40/40 [==============================] - 188s 5s/step - loss: 1.9032 - accuracy: 0.5117 - val_loss: 0.6617 - val_accuracy: 0.5417
Epoch 2/4
40/40 [==============================] - ETA: 0s - loss: 0.6313 - accuracy: 0.6342
Epoch 00002: val_accuracy improved from 0.54167 to 0.68750, saving model to NasNetLarge_features_balanced/tmp_model
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
40/40 [==============================] - 157s 4s/step - loss: 0.6313 - accuracy: 0.6342 - val_loss: 0.5792 - val_accuracy: 0.6875
Epoch 3/4
40/40 [==============================] - ETA: 0s - loss: 0.5577 - accuracy: 0.7110
Epoch 00003: val_accuracy did not improve from 0.68750
40/40 [==============================] - 154s 4s/step - loss: 0.5577 - accuracy: 0.7110 - val_loss: 0.5539 - val_accuracy: 0.6771
Epoch 4/4
40/40 [==============================] - ETA: 0s - loss: 0.5105 - accuracy: 0.7487
Epoch 00004: val_accuracy improved from 0.68750 to 0.71875, saving model to NasNetLarge_features_balanced/tmp_model
INFO:tensorflow:Assets written to: NasNetLarge_features_balanced/tmp_model/assets
40/40 [==============================] - 159s 4s/step - loss: 0.5105 - accuracy: 0.7487 - val_loss: 0.5376 - val_accuracy: 0.7188
