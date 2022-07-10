# Imports
import math
import os
import sys

import cv2 as cv
import numpy as np


def imagesToBW(folder):
    """
    Uses the same algorithm as boardscan.py to turn images into B/W images.

    Parameters:
    folder (string): the folder where your images are stored

    Returns:
    None
    """

    # For each image
    for piece_type in os.listdir(folder + "/training_data"):
        for data_type in ["/training_data/", "/testing_data/"]:
            for image in os.listdir(folder + data_type + piece_type):

                # Open the image
                filename = folder + data_type + piece_type + "/" + image
                src = cv.imread(filename, cv.CV_8UC1)

                # Convert it and overwrite it
                ret2, th2 = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                cv.imwrite(filename, th2)


if __name__ == "__main__":
    imagesToBW(os.path.dirname(os.path.realpath(__file__)) + "/dataset")
