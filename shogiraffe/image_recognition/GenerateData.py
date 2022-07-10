# Imports
import math
import os
import sys

import cv2 as cv
import numpy as np

# Absolute path used for images. It should be something like .../dataset
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/dataset"


def resize_all_images(new_size):
    """
    Resizes all images so that they are all square, with a width and height
      equal to new_size.

    Parameters:
    new_size (int): the new size of the images

    Returns:
    None
    """

    # For each piece image
    for piece_type in os.listdir(ROOT_FOLDER + "/training_data"):
        for data_type in ["/training_data/", "/testing_data/"]:
            for image in os.listdir(ROOT_FOLDER + data_type + piece_type):

                # Open the image
                filename = ROOT_FOLDER + data_type + piece_type + "/" + image
                src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

                # Resize it and overwrite it
                resized = cv.resize(src, (new_size, new_size))
                cv.imwrite(filename, resized)


def generate_all_translations(size, translation_amount, translation_step):
    """
    Modifies each image by moving it around a bit (vertically and horizontally).

    Parameters:
    size (int): the size of the images
    translation_amount (int) : the maximum number of pixels to move
    translation_step (int) : the step of the translation

    Returns:
    None
    """

    # For each image
    for piece_type in os.listdir(ROOT_FOLDER + "/training_data"):
        for data_type in ["/training_data/", "/testing_data/"]:
            for image in os.listdir(ROOT_FOLDER + data_type + piece_type):

                # Open the image
                filename = ROOT_FOLDER + data_type + piece_type + "/" + image
                src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

                # Translation
                for i in range(-translation_amount, translation_amount + 1, translation_step):
                    for j in range(-translation_amount, translation_amount + 1):
                        translation = cv.warpAffine(
                            src,
                            np.float32([[1, 0, i], [0, 1, j]]),
                            (size, size),
                            borderValue=(255, 255, 255),
                        )

                        # Write new images
                        cv.imwrite(
                            filename[:-4] + "tra" + str(i) + str(j) + ".png",
                            translation,
                        )


def generate_all_rotations(size, rotation_angle, rotation_step):
    """
    Modifies each image by turning it around a bit (around its center).

    Parameters:
    size (int): the size of the images
    rotation_angle (int) : the maximum number of degrees for the rotation
    rotation_step (int) : the step of the rotation, in degrees

    Returns:
    None
    """

    # For each image
    for piece_type in os.listdir(ROOT_FOLDER + "/training_data"):
        for data_type in ["/training_data/", "/testing_data/"]:
            for image in os.listdir(ROOT_FOLDER + data_type + piece_type):

                # Open the image
                filename = ROOT_FOLDER + data_type + piece_type + "/" + image
                src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

                # Rotation
                for i in range(-rotation_angle, rotation_angle + 1, rotation_step):
                    M = cv.getRotationMatrix2D((size / 2, size / 2), i, 1)
                    rotation = cv.warpAffine(src, M, (size, size), borderValue=(255, 255, 255))

                    # Write new images
                    cv.imwrite(filename[:-4] + "rot" + str(i) + ".png", rotation)


def rotate_images_180_degrees(size):
    """
    Modifies each image by turning it upside down.

    Parameters:
    size (int): the size of the images

    Returns:
    None
    """

    # For each image
    for piece_type in os.listdir(ROOT_FOLDER + "/training_data"):
        for data_type in ["/training_data/", "/testing_data/"]:

            # Create a new folder
            os.mkdir(ROOT_FOLDER + data_type + piece_type + "UD")
            for image in os.listdir(ROOT_FOLDER + data_type + piece_type):

                # Open the image
                filename = ROOT_FOLDER + data_type + piece_type + "/" + image
                src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

                # Rotate and write the new image
                M = cv.getRotationMatrix2D((size / 2, size / 2), 180, 1)
                rotation = cv.warpAffine(src, M, (size, size), borderValue=(255, 255, 255))
                cv.imwrite(ROOT_FOLDER + data_type + piece_type + "UD" + "/" + image, rotation)


# Depending on what you want to do, you may consider :

# resize_all_images(48)
# generate_all_translations(48, 6, 6)
# generate_all_rotations(48, 9, 6)
# rotate_images_180_degrees(48)
