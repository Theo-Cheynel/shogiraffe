# Imports
import sys
import math
import cv2 as cv
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# The root folder, used for loading weights
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))



def build_neural_network():
    """
    Builds our characteristic CNN with its regular layers.

    Parameters:
    None

    Returns:
    model (tensorflow.keras.models.Sequential): our NN model.
    """

    model = models.Sequential()

    model.add(layers.Conv2D(48, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    print(model.summary())

    return model



if __name__='__main__':

    # Build our CNN
    model = build_neural_network()

    # Compile it
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Load the weights obtained via training
    print("Loading weights...")
    checkpoint_dir = ROOT_FOLDER+"/dataset/checkpoints/cp.ckpt"
    model.load_weights(checkpoint_dir)

    # Test on pieces
    print("Testing on detected pieces...")

    filename = ROOT_FOLDER + '/dataset/testing_data/pawn/4ce0fe20f179cd5eb4e65ccae16099ee.rot3.png'
    squareij = cv.imread(filename, cv.IMREAD_GRAYSCALE)/255
    resized = cv.resize(squareij, (48, 48))
    resized = np.expand_dims(resized, axis=2)
    resized = np.expand_dims(resized, axis=0)
    
    print(model.predict(resized))
