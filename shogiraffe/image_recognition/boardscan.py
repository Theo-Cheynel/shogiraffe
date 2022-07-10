# Imports
import sys
import math
import cv2 as cv
import numpy as np
import os
import shogi

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Root folder, for reading the board image
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))

categories = ['king', 'kingUD', 'gold', 'goldUD', 'silver', 'silverUD',
              'knight', 'knightUD', 'lance', 'lanceUD', 'bishop', 'bishopUD',
              'rook', 'rookUD', 'pawn', 'pawnUD', 'empty']




def auto_canny(image, sigma=0.33):
    """
    Performs an automatic edge detection, without having to tune the parameters.

    Parameters:
    image (numpy array) : an image in B/W
    sigma (float, default 0.33) : the sigma-value of the algorithm. It defines
      the bounds around the image's median value.

    Returns:
    edged (numpy array) : the image after edge detection.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * 3*v))
    upper = int(min(255, (1.0 + sigma) * 3*v))
    edged = cv.Canny(image, lower, upper)

    # return the edged image
    return edged



def order_points(pts):
    """
    Given four points on the image, returns them in the right order so that they
    can be used without any problem by the four_point_transform method.
    They are returned ordered in the trigonometric way.

    Parameters:
    pts (quadruplet of (x,y) pairs) : the four points to analyze

    Returns:
    rect (quadruplet of (x,y) pairs) : the four points in the right order
    """
    # initialize a list of coordinates that will be ordered such that the first
    # entry in the list is the top-left, the second entry is the top-right,
    # the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas the bottom-right point
    # will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the top-right point will have
    # the smallest difference, whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """
    Given four points on the image, crops the image to the delimited zone.
    This is used to correct perspective deformations.

    Parameters:
    image (numpy array) : an image to modify
    pts (quadruplet of (x,y) pairs) : the four points to analyze

    Returns:
    warped (numpy array) : the image after perspective correction
    """
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the maximum distance between
    # bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the maximum distance between
    # the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped



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
    model.add(layers.Dense(17, activation='softmax'))

    return model



def recadrer_image(filename, showResults=False, writeSquaresAsImages=False):
    """
    Performs an automatic cropping of the image to center on the board.
    It will first switch the image to black and white using Otsu's binarization,
    then it will perform various treatments before using Canny Edge detection
    to crop the image along the edges of the board.

    Parameters:
    filename (string) : the filename of the image to process
    showResults (boolean, default False) : if True, displays an image after
      every operation (useful for debugging and explaining)
    writeSquaresAsImages (boolean, default False) : if True, will save new images
      for every square on the board (useful when building a dataset)

    Returns:
    squareResize (numpy array) : the image after being cropped
    """

    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    # This will be used for displaying the images, only if showResults is True
    scale_percent = 80 # percent of original size
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Show the results so far
    if showResults :
        cv.imshow("Source", cv.resize(src, dim))
        cv.waitKey()


    # Image to B/W
    # This uses an adaptive gaussian threshold as a mask (it eliminates noise),
    # but the principal processing is made with Otsu's binarization.
    ret, mask = cv.threshold(src, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # The mask of the image
    th2 = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv.THRESH_BINARY,17,7)

    # Multiply the result by the mask
    th2 = np.uint8(255-np.multiply(255-th2, 1-mask/255))


    # Show the results so far
    if showResults :
        cv.imshow("Thresholding", th2)
        cv.waitKey()


    # Perform a simple convolution for the noise that is left
    kernel = np.ones((2,2),np.uint8)
    closing = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel)

    # Show the results so far
    if showResults :
        cv.imshow("Closure", closing)
        cv.waitKey()


    # Automatic edge detection
    dst = auto_canny(closing)

    # Show the results so far
    if showResults :
        cv.imshow("Edge detection", dst)
        cv.waitKey()


    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # Line detection with probabilistic Hough transform
    linesP = cv.HoughLinesP(dst, 2, np.pi / 360, 75, None, 40, 35)

    if linesP is not None:
        for i in range(0, len(linesP)):
            # For each line, look at the angle (not used yet, maybe in the future)
            l = linesP[i][0]
            angle = (math.atan2((l[1]-l[3]),(l[0]-l[2]))*180/math.pi) % 180
            if angle > 150 :
                angle = angle - 180
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    # Show the results so far
    if showResults :
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cv.resize(cdstP, dim))
        cv.waitKey()


    # Crop the image based on the line detection
    # To do this, we have to find the "corner points" of the board, i.e., the
    # points that are in the top-left, bottom-left, top-right, bottom-right.
    if linesP is not None:
        leftupmost = None
        rightupmost = None
        leftdownmost = None
        rightdownmost = None

        upleft = np.inf
        upright = -1
        downleft = -1
        downright = -1

        for i in range(0, len(linesP)):
            line = linesP[i][0]

            if line[0] + line[1] < upleft :
                leftupmost = (line[0], line[1])
                upleft = line[0] + line[1]

            if line[0] + line[1] > downright :
                rightdownmost = (line[0], line[1])
                downright = line[0] + line[1]

            if line[0] - line[1] > upright :
                rightupmost = (line[0], line[1])
                upright = line[0] - line[1]

            if line[1] - line[0] > downleft :
                leftdownmost = (line[0], line[1])
                downleft = line[1] - line[0]


            if line[2] + line[3] < upleft :
                leftupmost = (line[2], line[3])
                upleft = line[2] + line[3]

            if line[2] + line[3] > downright :
                rightdownmost = (line[2], line[3])
                downright = line[2] + line[3]

            if line[2] - line[3] > upright :
                rightupmost = (line[2], line[3])
                upright = line[2] - line[3]

            if line[3] - line[2] > downleft :
                leftdownmost = (line[2], line[3])
                downleft = line[3] - line[2]

        # Crop the image with these four points, accounting for perspective
        b = four_point_transform(th2, np.array([leftupmost, rightupmost, rightdownmost, leftdownmost]))

    # Resize the image so that it becomes square
    size = max(int(b.shape[1] * scale_percent / 100), int(b.shape[0] * scale_percent / 100))
    squareResize = cv.resize(b, (size, size))

    # Show the results so far
    if showResults :
        cv.imshow("Recadrage", squareResize)
        cv.waitKey()

    # If needed, write every square in memory (useful when building dataset)
    if writeSquaresAsImages :
        k = size / 9

        # For each square, we'll write it in storage using a different name
        for i in range(9):
            for j in range(9) :
                squareij = squareResize[int(i*k+3):int(i*k+k-3), int(j*k+3):int(j*k+k-3)]
                cv.imwrite(str(i+1)+str(j+1)+"c.png", squareij)


    # Return the numpy array of the resized image
    return squareResize




def analyser_image(image):
    """
    Uses our pre-trained CNN to classify each square of the board.

    Parameters:
    image (numpy array) : the resized image

    Returns:
    board (shogi.Board) : the board after image processing and recognition.
    """

    # Create a basic model instance
    model = build_neural_network()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Load the weights of our pre-trained model
    checkpoint_dir = ROOT_FOLDER+"/dataset/checkpoints/cp.ckpt"
    model.load_weights(checkpoint_dir)

    # Get the dimensions of the image
    width, height = image.shape[:2]

    k = width / 9
    s = ""

    for i in range(9):
        for j in range(9) :
            # For each square of the board
            squareij = image[int(i*k+3):int(i*k+k-3), int(j*k+3):int(j*k+k-3)]

            resized = cv.resize(squareij, (48, 48))
            resized = np.expand_dims(resized, axis=2)
            resized = np.expand_dims(resized, axis=0)

            # Feed it in the neural network
            result = model.predict(resized)
            piece = categories[np.argmax(result)]

            # Add it to the "SFEN" representation of the board
            if piece == "empty" :
                if len(s)>0 and s[-1] in "12345678" :
                    s = s[:-1] + str(int(s[-1])+1)
                    r=""
                else :
                    r="1"
            elif piece == "knight" :
                r = "N"
            elif piece == "knightUD" :
                r = "n"
            elif piece[-2:] == "UD" :
                r = piece[0]
            else :
                r = piece[0].capitalize()
            s += r
        s+="/"

    # Print the board's SFEN representation
    print(s[:-1] + " b")

    # Return the board
    return shogi.Board(s[:-1] + " b - 1")



if __name__ == "__main__":

    # An example board
    filename = ROOT_FOLDER + "/board_images/board4.JPG"

    # Crop the image on the board
    img = recadrer_image(filename, True, False)

    # Process the image and print the board after image recognition
    print(analyser_image(img))
