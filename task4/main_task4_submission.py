import numpy as np
import os
import time
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Random states
random_state = 4
np.random.seed(random_state)
tf.random.set_seed(random_state)

"""
    -- IML Project Task 4 Code by Daniel Kiesewalter and Leiv Andresen --
    - execution requirements below -

    - Task description:

    The task is to reason about the relative taste of food on images. More precisely we are presented
    with three images of food, A, B and C, and we need to say if the food in image A tastes more like the
    food in image B or C. 

    - Approach description: 

    In the toturials we were advised to follow a strategy of transfer learning. That is to build on top of a 
    model that has been developed, trained and extensively optimized on other (similar) datasets. 
    As we are dealing with image classification, good models consist of a Convolutional Neural Networks
    to extract relevant information (features) from the images followed by a comparatively small fully 
    connected network to predict classes based on the features. 

    We decided to use the model NasNetLarge as feature extractor and train a dense neural net on top to perform
    the classification. The images are first preprocessed (scaling piel values to -1, 1) before the NasNetLarge 
    model predicts the final layer before classification. To reduce the dimensionality global average pooling is
    applied to this layer yielding roughly 4000 feature values per image. The NasNetLarge model predicts with 
    weights pretrained on the 'ImageNet' dataset.

    Thereafter image B and C are swapped for half of the provided training image triplets in order to generate 
    a balanced training dataset. Before training the data is split into unique train and validation sets in order to 
    ensure that the model generalizes well to unseen images since the submission test set contains images that 
    are not in the training set. This split yields a validation set of roughly 100 images and a training set
    of roughly 40 000 images. 

    The training set is fed into a dense neural net with a 'relu' input layer two 'relu' hidden layers and a
    'sigmoid' output layer. Thereafter the output is thresholded at 0.5 to determine the class prediction.
    The optimizer is 'adam' and the loss 'binary_crossentropy'. When iterating hyperparameters we noticed that we 
    needed to adjust the batch size so the net does not overfit to the training set already after the first epoch. 
    In our final setup the best validation score is still achieved in the first few epochs.

    - Execution requirements:

    Tested only on Python version 3.7
    modules: sklearn, pandas, numpy, time, os
    pickle is used to save/load objects: https://docs.python.org/3/library/pickle.html
    For the download of the NasNetLarge model to work we had to follow the first answer here: 
    https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664
    and then install pillow: 
    https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664

    "train_triplets.txt", "test_triplets.txt" and the folder "food" with the images have to be located in the
    same folder as the script. The script will create and save files to the folder "NasNetLarge_features_balanced".

    The total space required is roughly 25 GB and the time required is around 6 hours on our machine.
    The results will be saved to the folder in which the script is placed.
    If the script is rerun then it will load the features from the disk.
    If you have problems feel free to contact us at leiva@ethz.ch.

    """


def extract_features_from_images():
    # Check if images have already been extracted
    try:
        feature_list = pickle.load(open(os.path.join(
            dir_name, "image_features_NasNetLarge_avg_pooling.p"), "rb"))
        image_name_list = pickle.load(open(os.path.join(
            dir_name, "image_list_NasNetLarge_avg_pooling.p"), "rb"))
        print("[extract_features_from_images] Successfully loaded preexisting NasNetLarge features.")
        return feature_list, image_name_list
    except FileNotFoundError:
        # Extract features from images
        print("[extract_features_from_images] Extracting NasNetLarge features...")
        # Load images from directory
        image_directory = "food"
        image_list = os.listdir(image_directory)
        feature_list = []
        image_name_list = []

        # Load ResNet50
        model_notop = NASNetLarge(weights='imagenet', include_top=False, pooling='avg')

        # Extract features from images
        start = time.time()
        counter = 0
        for image_name in image_list:
            try:
                # Load image and preprocess
                img = image.load_img(os.path.join(image_directory, image_name), target_size=(331, 331))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = tf.keras.applications.nasnet.preprocess_input(x, data_format=None)

                # Compute features
                features = model_notop.predict(x)
                feature_list.append(features)
                image_name_list.append(image_name)

                if counter % 500 == 0:
                    print(f"[extract_features_from_images] {counter} images extracted")
                    pickle.dump(feature_list, open(os.path.join(
                                    dir_name, "image_features_NasNetLarge_avg_pooling.p"), "wb"))
                    pickle.dump(image_name_list, open(os.path.join(
                                    dir_name, "image_list_NasNetLarge_avg_pooling.p"), "wb"))
                counter += 1

            # handles the odd operating system helper files in the folder
            except:
                print(f"[extract_features_from_images] ERROR: with image name {image_name}")

        print(f"[extract_features_from_images] Processing images took: {'%.2f' % ((time.time() - start)/60)} minutes")

        # File I/O
        pickle.dump(feature_list, open(os.path.join(
                                    dir_name, "image_features_NasNetLarge_avg_pooling.p"), "wb"))
        pickle.dump(image_name_list, open(os.path.join(
                                    dir_name, "image_list_NasNetLarge_avg_pooling.p"), "wb"))
        feature_list = pickle.load(open(os.path.join(
                                    dir_name, "image_features_NasNetLarge_avg_pooling.p"), "rb"))
        image_name_list = pickle.load(open(os.path.join(
                                    dir_name, "image_list_NasNetLarge_avg_pooling.p"), "rb"))

        return feature_list, image_name_list


def create_feature_vectors():
    # Try to load preexisting data
    try:
        train_features = np.load(os.path.join(
            dir_name, "x_train_Resnet50_avg_pool_balanced__all_features.npy"))
        # test_features = pd.read_csv("x_test_Resnet50_avg_pool.csv").iloc[:, 1:]
        train_triplets_balanced = pd.read_csv(os.path.join(
            dir_name, "train_triplets_balanced_w_label.csv")).iloc[:, 1:]
        print("[create_feature_vectors] Loaded preexisting feature vectors.")
        return train_features, train_triplets_balanced
    except FileNotFoundError:
        print("[create_feature_vectors] Creating feature vectors...")

    # Truncate .jpg off of list of images
    list_of_image_names = [os.path.splitext(image_filename)[0] for image_filename in list_of_images]

    # Create random indexes
    random_indexes = np.random.choice(train_triplets.index,
                                      size=int(len(train_triplets.index) / 2),
                                      replace=False)
    # switch image B and C for half the training set randomly to have an balanced dataset of 1 and 0 classifications
    train_triplets_balanced = pd.read_table("train_triplets.txt", names=['A', 'B', 'C'], delimiter=' ', dtype=str)
    train_triplets_balanced.iloc[random_indexes, 1] = train_triplets.iloc[random_indexes, 2]
    train_triplets_balanced.iloc[random_indexes, 2] = train_triplets.iloc[random_indexes, 1]
    train_labels = pd.DataFrame(np.ones((len(train_triplets.index), 1), dtype=int))
    train_labels.iloc[random_indexes] = 0

    # Create train features dataframe
    train_features_temp = pd.DataFrame()
    batch = int(0)
    print("[create_feature_vectors] Training features...")
    for index, row in train_triplets_balanced.iterrows():
        # index = list_of_image_names.index(row['A']) # Find the index of an imagename in the feature list
        # list_of_features[index] # get the feature at that index
        features = np.concatenate((
            list_of_features[list_of_image_names.index(row['A'])].reshape(-1),
            list_of_features[list_of_image_names.index(row['B'])].reshape(-1),
            list_of_features[list_of_image_names.index(row['C'])].reshape(-1)
        ))
        # The following line takes at least 40% of the execution time
        train_features_temp = train_features_temp.append(pd.DataFrame(features.reshape(1, -1)))
        if index % 6000 == 0 and index != 0 or index == len(train_triplets_balanced.index) - 1:
            print(index / (len(train_triplets_balanced.index) - 1), "Batch", batch)
            train_features_temp.to_csv(os.path.join(
                dir_name, ("x_train_NasNetLarge_avg_pool_balanced_" + str(batch) + ".csv")))
            del train_features_temp
            train_features_temp = pd.DataFrame()
            batch += 1

    # Save labels
    train_triplets_balanced.insert(0, "label", train_labels, True)
    train_triplets_balanced.to_csv(os.path.join(
        dir_name,
        "train_triplets_balanced_w_label.csv"))
    print("[create_feature_vectors] Training features done.")

    # Create test features dataframe
    test_features = pd.DataFrame()
    batch = int(0)
    print("[create_feature_vectors] Test features...")
    for index, row in test_triplets.iterrows():
        # index = list_of_image_names.index(row['A']) # Find the index of an imagename in the feature list
        # list_of_features[index] # get the feature at that index, reshape(-1) to make a 1d array
        features = np.concatenate((
            list_of_features[list_of_image_names.index(row['A'])].reshape(-1),
            list_of_features[list_of_image_names.index(row['B'])].reshape(-1),
            list_of_features[list_of_image_names.index(row['C'])].reshape(-1)
        ))
        test_features = test_features.append(pd.DataFrame(features.reshape(1, -1)))
        if (index % 6000 == 0 and index != 0) or index == len(test_triplets.index) - 1:
            print(index / (len(test_triplets.index) - 1), "Batch", batch)
            test_features.to_csv(os.path.join(
                dir_name, ("x_test_NasNetLarge_avg_pool_" + str(batch) + ".csv")))
            del test_features
            test_features = pd.DataFrame()
            batch += 1

    print("[create_feature_vectors] Test features done.")

    # Piece together feature vectors
    test_features = piece_together_feature_batches(dir_name, "x_test_NasNetLarge_avg_pool_")
    del test_features
    train_features = piece_together_feature_batches(dir_name, "x_train_NasNetLarge_avg_pool_balanced_")

    return train_features, train_triplets_balanced


def piece_together_feature_batches(directory_name, file_name):
    all_features = pd.read_csv(os.path.join(directory_name, (file_name + str(0) + ".csv"))).iloc[:, 1:].to_numpy()
    batch = int(1)
    try:
        while True:
            current_batch = pd.read_csv(os.path.join(directory_name, (file_name + str(batch) + ".csv"))).iloc[:, 1:]
            current_batch_np = current_batch.to_numpy()
            all_features = np.concatenate((all_features, current_batch), axis=0)
            print(f"Batch {batch} appended. Batch size: {current_batch_np.shape}, Total size: {all_features.shape}")
            del current_batch
            del current_batch_np
            batch += 1
    except FileNotFoundError:
        print(f"[piece_together_feature_batches] {file_name} batches finished or no files found.")

    np.save(os.path.join(directory_name, (file_name + "all_features")), all_features)
    return all_features


def train_val_split(train_triplets_, threshold: int):
    # Creates a test and validation split with strictly distinct images. This may lead to loss of data, as it might be
    # impossible to find a perfect partition i.e. there can be tuples that contain photos above and below our threshold
    greater_thresh = train_triplets_.astype(int) > threshold
    greater_thresh = greater_thresh.sum(axis=1).astype(bool)
    def_smaller_index = greater_thresh.index[greater_thresh == False].tolist()
    val = train_triplets_.loc[def_smaller_index]
    smaller_thresh = train_triplets_.astype(int) < threshold
    smaller_thresh = smaller_thresh.sum(axis=1).astype(bool)
    def_greater_index = smaller_thresh.index[smaller_thresh == False].tolist()
    train = train_triplets_.loc[def_greater_index]
    print('Datapoints lost during distinct test-train splitting %s' % (len(train_triplets_.index) - len(val.index) - len(train.index)))
    # Check for no duplicates
    assert(pd.merge(val, train, how='inner').size == 0)
    print(f"Validation set size {val.shape}")
    print(f"Training set size {train.shape}")
    return train, val


if __name__ == '__main__':
    # Make working directory if it does not exist
    dir_name = "NasNetLarge_features_balanced"
    os.makedirs(dir_name, exist_ok=True)

    # Load ResNet50 features from preexisting data or create if they don't exist
    try:
        x_train = np.load(os.path.join(
            dir_name, "x_train_NasNetLarge_avg_pool_balanced_all_features.npy"))
        train_triplets_balanced_with_y = pd.read_csv(os.path.join(
            dir_name, "train_triplets_balanced_w_label.csv")).iloc[:, 1:]
        print("Loaded preexisting feature vectors.")
    except FileNotFoundError:
        print("[create_feature_vectors] Creating feature vectors...")
        # Extract features from images
        list_of_features, list_of_images = extract_features_from_images()
        # Load traning and test triplets
        train_triplets = pd.read_table("train_triplets.txt", names=['A', 'B', 'C'], delimiter=' ', dtype=str)
        test_triplets = pd.read_table("test_triplets.txt", names=['A', 'B', 'C'], delimiter=' ', dtype=str)
        # Create feature vectors
        x_train, train_triplets_balanced_with_y = create_feature_vectors()
        # Free up memory
        del list_of_features
        del list_of_images

    # Extract labels
    y_train = train_triplets_balanced_with_y.iloc[:, 0].to_numpy()

    # Build Neural Net
    model = Sequential()
    model.add(Dense(np.shape(x_train)[1], input_dim=np.shape(x_train)[1], activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Split training data into a validation and test set that do not contain common images
    train_set, val_set = train_val_split(train_triplets_balanced_with_y.iloc[:, 1:], threshold=620)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Starting training of the model ...")
    filepath = os.path.join(dir_name, "tmp_model")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    model.fit(x_train[train_set.index, :], y_train[train_set.index], epochs=8, batch_size=1024, shuffle=True,
              validation_data=(x_train[val_set.index, :], y_train[val_set.index]),
              callbacks=[checkpoint])

    # Reload best weights
    model = tf.keras.models.load_model(filepath)
    # Predict for submission set
    del x_train
    X_submission = np.load(os.path.join(
            dir_name, "x_test_NasNetLarge_avg_pool_all_features.npy"))
    y_submission = model.predict(X_submission)
    y_submission[y_submission > 0.5] = int(1)
    y_submission[y_submission < 0.5] = int(0)
    np.savetxt("NasNetLarge_small_net_split_620_batch_1024_val3", y_submission.astype(dtype=int), fmt='%d')
