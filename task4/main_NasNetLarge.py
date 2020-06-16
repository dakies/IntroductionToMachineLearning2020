import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import pandas as pd
import pprofile
# from livelossplot import PlotLossesKeras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

""" Task4 - Classify image similar in taste 
    
    Strategy: Extract features using transfer learning and ResNet50, make classifier therafter.
"""

# Random states
random_state = 4
np.random.seed(random_state)


def checkfeature(image_name):
    # Load images from directory
    image_directory = "food"
    image_list = os.listdir(image_directory)
    feature_list = []
    image_name_list = []

    # Load ResNet50
    model_notop = NASNetLarge(weights='imagenet', include_top=False, pooling='avg')
    # Load image and preprocess
    img = image.load_img(os.path.join(image_directory, image_name), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # Compute features
    features = model_notop.predict(x)

    # Todo:change these vals
    diff = x_train[2, :2048] - features
    print(diff.max())


def extract_features_from_images():
    # Check if images have already been extracted
    try:
        feature_list = pickle.load(open("NasNetLarge_features_balanced/image_features_NasNetLarge_avg_pooling.p", "rb"))
        image_name_list = pickle.load(open("NasNetLarge_features_balanced/image_list_NasNetLarge_avg_pooling.p", "rb"))
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
                    pickle.dump(feature_list,
                                open("NasNetLarge_features_balanced/image_features_NasNetLarge_avg_pooling.p", "wb"))
                    pickle.dump(image_name_list,
                                open("NasNetLarge_features_balanced/image_list_NasNetLarge_avg_pooling.p", "wb"))
                counter += 1

            # handles the odd operating system helper files in the folder TODO make this safer
            except:
                print(f"[extract_features_from_images] ERROR: with image name {image_name}")

        print(f"[extract_features_from_images] Processing images took: {'%.2f' % (time.time() - start)} seconds")

        # File I/O
        pickle.dump(feature_list, open("NasNetLarge_features_balanced/image_features_NasNetLarge_avg_pooling.p", "wb"))
        pickle.dump(image_name_list, open("NasNetLarge_features_balanced/image_list_NasNetLarge_avg_pooling.p", "wb"))
        feature_list = pickle.load(open("NasNetLarge_features_balanced/image_features_NasNetLarge_avg_pooling.p", "rb"))
        image_name_list = pickle.load(open("NasNetLarge_features_balanced/image_list_NasNetLarge_avg_pooling.p", "rb"))

        return feature_list, image_name_list


def create_feature_vectors():
    # Prepare directory
    dir_name = "NasNetLarge_features_balanced"
    os.makedirs(dir_name, exist_ok=True)

    # Try to load preexisting data
    try:
        train_features = np.load(os.path.join(dir_name,
                                              "x_train_Resnet50_avg_pool_balanced__all_features.npy"))
        # test_features = pd.read_csv("x_test_Resnet50_avg_pool.csv").iloc[:, 1:]
        train_triplets_balanced = pd.read_csv(
            os.path.join(dir_name,
                         "train_triplets_balanced_w_label.csv")).iloc[:, 1:]
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
                dir_name,
                ("x_train_NasNetLarge_avg_pool_balanced_" + str(batch) + ".csv")))
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
                dir_name,
                ("x_test_NasNetLarge_avg_pool_" + str(batch) + ".csv")))
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


if __name__ == '__main__':
    # Load ResNet50 features from preexisting data or create if they don't exist
    try:
        # TODO Insert path to .npy and .csv files
        x_train = np.load("NasNetLarge_features_balanced/x_train_NasNetLarge_avg_pool_balanced__all_features.npy")
        train_triplets_balanced_with_y = pd.read_csv(
            "NasNetLarge_features_balanced/train_triplets_balanced_w_label.csv").iloc[:, 1:]
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

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Starting training of the model ...")
    model.fit(x_train, y_train, validation_split=0.33, epochs=15, batch_size=256, shuffle=True)

    # Predict for submission set
    del x_train
    # TODO Insert path to .npy file
    X_submission = np.load("NasNetLarge_features_balanced/x_test_NasNetLarge_avg_pool_all_features.npy")
    y_submission = model.predict(X_submission)
    y_submission[y_submission > 0.5] = int(1)
    y_submission[y_submission < 0.5] = int(0)
    # TODO Change name of output so the old one isn't overwritten
    np.savetxt("NasNetLarge_vanilla_dense_test", y_submission.astype(dtype=int), fmt='%d')

    # ---SKLEARN---
    # # Train test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     x_train, y_train, test_size=0.5, random_state=random_state, stratify=y_train)

    # print("Fitting models")
    # models = {
    #     # "Support Vector Classifier": SVC(verbose=1, random_state=random_state),
    #     "Linear Support Vector Classifier": LinearSVC(verbose=1, random_state=random_state, max_iter=300),
    #     "Multilayer Perceptron Classifier": MLPClassifier(random_state=random_state, verbose=1)
    # }
    #
    # # Train and predict for models
    # for model_name, model in models.items():
    #     print("Training", model_name, "...")
    #     model.fit(X_train, y_train)
    #     y_test_hat = model.predict(X_test)
    #     print("Accuracy score of", model_name, accuracy_score(y_true=y_test, y_pred=y_test_hat))
    #
    # # Evaluate
    # for model_name, model in models.items():
    #     y_test_hat = model.predict(X_test)
    #     print("Accuracy score of", model_name, accuracy_score(y_true=y_test, y_pred=y_test_hat))
