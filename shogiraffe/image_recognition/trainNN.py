# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import math
import os
import random
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, datasets, layers, models

# The root folder, used for saving weights and reading images
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/dataset"

categories = [
    "king",
    "kingUD",
    "gold",
    "goldUD",
    "silver",
    "silverUD",
    "knight",
    "knightUD",
    "lance",
    "lanceUD",
    "bishop",
    "bishopUD",
    "rook",
    "rookUD",
    "pawn",
    "pawnUD",
    "empty",
]


def images_to_numpy():
    """
    Loads and changes the format of the data so that they are ready to be fed into our NN.

    Parameters:
    None

    Returns:
    (train_images, train_labels, test_images, test_labels) where :
      train_images is a numpy 4D-vector representing our training data
      train_labels is a numpy 3D-vector representing our training labels
      test_images is a numpy 4D-vector representing our testing data
      test_labels is a numpy 3D-vector representing our testing labels

    """

    train_images, train_labels, test_images, test_labels = [], [], [], []

    for cat in categories:
        print("Now loading category :'" + cat + "'")

        # Load the training data for given category
        files = os.listdir(ROOT_FOLDER + "/training_data/" + cat)
        random.shuffle(files)

        # Add each image to the training data
        for f in files:
            img = cv.imread(ROOT_FOLDER + "/training_data/" + cat + "/" + f, cv.IMREAD_GRAYSCALE) / 255
            train_images.append(img)
            train_labels.append(categories.index(cat))

        # Load the testing data for given category
        files = os.listdir(ROOT_FOLDER + "/testing_data/" + cat)
        random.shuffle(files)

        # Add each image to the testing data
        for f in files:
            img = cv.imread(ROOT_FOLDER + "/testing_data/" + cat + "/" + f, cv.IMREAD_GRAYSCALE) / 255
            test_images.append(img)
            test_labels.append(categories.index(cat))

    # Shuffle the training data
    rd = list(zip(train_images, train_labels))
    random.shuffle(rd)
    train_images, train_labels = zip(*rd)

    # Shuffle the testing data
    rd = list(zip(test_images, test_labels))
    random.shuffle(rd)
    test_images, test_labels = zip(*rd)

    # Since the images are in B/W, they are 2D arrays, but we need 3D arrays
    train_images = np.expand_dims(np.array(train_images), axis=3)
    train_labels = np.expand_dims(np.array(train_labels), axis=1)
    test_images = np.expand_dims(np.array(test_images), axis=3)
    test_labels = np.expand_dims(np.array(test_labels), axis=1)

    # Return our properly formatted dataset, ready for training
    return (train_images, train_labels, test_images, test_labels)


def build_neural_network():
    """
    Builds our characteristic CNN with its regular layers.

    Parameters:
    None

    Returns:
    model (tensorflow.keras.models.Sequential): our NN model.
    """

    model = models.Sequential()

    model.add(layers.Conv2D(48, (3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(17, activation="softmax"))

    print(model.summary())

    return model


def train_neural_network(model, data):
    """
    Builds our characteristic CNN with its regular layers.

    Parameters:
    model (tensorflow.keras.models.Sequential): the model to train
    data (quadruplet of numpy arrays) : the data on which the model should be trained

    Returns:
    model (tensorflow.keras.models.Sequential): our NN model after training.
    """

    # Separate the datasets
    (train_images, train_labels, test_images, test_labels) = data

    # The path for the checkpoint
    checkpoint_path = ROOT_FOLDER + "/checkpoints/cp.ckpt"

    # The callback for model saving
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Compile our CNN
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Start training
    history = model.fit(
        train_images,
        train_labels,
        epochs=5,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback],
    )

    # Returned our trained model
    return model


data = images_to_numpy()
print("Image dataset successfully loaded !")
nn = build_neural_network()
train_neural_network(nn, data)
