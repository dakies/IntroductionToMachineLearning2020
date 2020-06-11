import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from sklearn.neural_network import MLPClassifier

""" Task4 - Classify image similar in taste 
    
    Strategy: Extract features using transfer learning and ResNet50, make classifier therafter.
"""


def extract_features_from_images():
    # Check if images have already been extracted
    try:
        feature_list = pickle.load(open("image_features_ResNet50.p", "rb"))
        image_name_list = pickle.load(open("image_list_ResNet50.p", "rb"))
        print("[extract_features_from_images] Successfully loaded preexisting ResNet50 features.")
        return feature_list, image_name_list
    except FileNotFoundError:
        # Extract features from images
        print("[extract_features_from_images] Extracting ResNet50 features...")
        # Load images from directory
        image_directory = "food"
        image_list = os.listdir(image_directory)
        feature_list = []
        image_name_list = []

        # Load ResNet50
        model_notop = ResNet50(weights='imagenet', include_top=False)

        # Extract features from images
        start = time.time()
        for image_name in image_list:
            try:
                # Load image and preprocess
                img = image.load_img(os.path.join(image_directory, image_name), target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # Compute features
                features = model_notop.predict(x)
                feature_list.append(features)
                image_name_list.append(image_name)

            # handles the odd operating system helper files in the folder TODO make this safer
            except:
                print(f"[extract_features_from_images] ERROR: with image name {image_name}")

        print(f"[extract_features_from_images] Processing images took: {'%.2f' % (time.time() - start)} seconds")

        # File I/O
        pickle.dump(feature_list, open("image_features_ResNet50.p", "wb"))
        pickle.dump(image_name_list, open("image_list_ResNet50.p", "wb"))
        feature_list = pickle.load(open("image_features_ResNet50.p", "rb"))
        image_name_list = pickle.load(open("image_list_ResNet50.p", "rb"))

        return feature_list, image_name_list


if __name__ == '__main__':
    # Extract features from images
    list_of_features, list_of_images = extract_features_from_images()

    # Load traning and test triplets
    train_triplets = pd.read_table("train_triplets.txt", names=['A', 'B', 'C'], delimiter=' ')
    test_triplets = pd.read_table("test_triplets.txt", names=['A', 'B', 'C'], delimiter=' ')

    # TODO: Find out how to create the training/test data:
    # -- switch image B and C for half the training set randomly to have an balanced dataset of 1 and 0 classifications
    # -- the encoding of the images from ResNet50 has shape (1, 7, 7, 2048), (huge!), flatten into a 1D array?
    # --- there are also ways to modify the encoding such that a pooling is applied at the end to give a lower
    # --- dimensional encoding that might be more tractable



