""" 
    ####Image Segmentation class###
    @author: Fraol Gelana
    @Institute:Artificial Intelligence Center
    @Date:December,2020
"""


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from Classifier.Classify import Classify
from Classifier.EvaluateSegment import Evaluator


class ImageSegment:
    image_path = ''  # Image path variable
    leafArea = None  # Total leaf Area
    fileName = None
    start_time = None
    stop_time = None

    def __init__(self, image_path, model_path, filename):
        self.image_path = image_path  # load image path
        self.model_path = model_path  # load saved deep learning model path
        self.fileName = filename
        self.currentMeanArray = []      # Empty array to hold the calculated Mean,
        self.currentVarianceArray = []  # Variance and
        self.areaUnderArray = []        # area under the mean-variance line
        self.readImage(self.image_path)  # read image

        return

    def readImage(self, image_path):
        self.start_time = time.time()

        leaf_image = cv.imread(image_path)

        self.remove_background(leaf_image)

        return

        # OpenCV reads all images in BGR color space by default

    def remove_background(self, leaf_image):

        # Gaussian blur image to remove noise
        blured = cv.GaussianBlur(leaf_image, (1, 1), 0)

        # Convert blured Image from BGR to HSV
        hsv_leaf = cv.cvtColor(blured, cv.COLOR_BGR2HSV)

        SV_channel = hsv_leaf.copy()

        SV_channel[:, :, 0] = np.zeros(
            (SV_channel.shape[0], SV_channel.shape[1]))  # Set the 'H' channel to Zero

        SV_channel[:, :, 2] = np.zeros(
            (SV_channel.shape[0], SV_channel.shape[1]))
        # Create a binary mask from the SV Channel

        mask = cv.inRange(SV_channel, (0, 0, 0), (0, 110, 0))
        # Invert mask, White areas represent green components and black the background
        mask = cv.bitwise_not(mask)

        # perform bitwise_and between mask and hsv image

        background_extracted = cv.bitwise_and(hsv_leaf, hsv_leaf, mask=mask)

        # calculate the contour area to find the total area of the leaf
        contours, heirarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largestContour = max(contours, key=cv.contourArea)

        self.leafArea = int(cv.contourArea(largestContour))

        self.image_correction(background_extracted, leaf_image, self.leafArea)

        return

    def image_correction(self, hsv_image, leaf_image, leafArea):

        rgb_image_equ = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)

        self.color_segment(hsv_image, rgb_image_equ, leaf_image, leafArea)

        return

    def color_segment(self, hsv_space, rgb_space, leaf_image, leafArea, lowerB=(36, 0, 0), upperB=(65, 255, 255), count=2):
        # extracted in the HSV color space

        # create binary mask using the bounds
        mask = cv.inRange(hsv_space, lowerB, upperB)
        mask = cv.bitwise_not(mask)

        # bitwise_and mask and rgb image
        output_hsv = cv.bitwise_and(hsv_space, hsv_space, mask=mask)
        output_rgb = cv.bitwise_and(rgb_space, rgb_space, mask=mask)

        nonZeroIntentsity = output_hsv[:, :, 0].copy()

        # Extract intensity values between [3,65]
        # This step ensures the black(intensity 0 - 2) pixel values
        # from the mean calculation.
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity > 3]
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity < 66]

        l = self.findLowerBound(nonZeroIntentsity)
        # Update the lower bound value of the 'H' channel accordingly
        # (H,S,V)
        new_lowerB = (l, 0, 0)

        mask = cv.inRange(hsv_space, new_lowerB, upperB)
        mask = cv.bitwise_not(mask)

        # bitwise_and mask and rgb image
        o_rgb = cv.bitwise_and(output_rgb, output_rgb, mask=mask)

        self.thresh_mask(o_rgb, leaf_image, leafArea)

        return

    def findLowerBound(self, intensityArray):

        nonZeroIntentsity = intensityArray

        # Find the mean and variance of the nonzero
        # intensity array
        mean = int(np.mean(nonZeroIntentsity))
        variance = int(np.var(nonZeroIntentsity))

        # Add the calculated mean and variance into
        # currentMeanArray and currentVarianceArray,respectively
        self.currentMeanArray.append(mean)
        self.currentVarianceArray.append(variance)

        # Update the nonZeroIntensity array,according to the current
        # mean value
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity <= mean]
        # Calculate the area of the right-triangle formed by the value of the Variance
        # and the value of the Mean
        #           |\
        #           | \
        #   Variance|  \
        #           |   \
        #           |____\
        #            Mean
        self.areaUnderArray.append(
            (self.currentMeanArray[-1] * self.currentVarianceArray[-1]) / 2)

        if len(self.areaUnderArray) >= 2 and self.areaUnderArray[-2] >= self.areaUnderArray[-1]:

            return self.currentMeanArray[-2]
        else:
            lowerBound = self.findLowerBound(nonZeroIntentsity)
            return lowerBound

    def thresh_mask(self, out, leaf_image, leafArea):
        mask = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
        # Calculate the otsu threshold
        ret, thresh = cv.threshold(
            mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Morphological close operation
        kernel = np.ones((3, 3))

        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        time_taken = time.time() - self.start_time
        # print(time_taken)

        Evaluator(thresh, self.fileName, time_taken)

        #self.find_contours(thresh, out, leaf_image, leafArea)

    def find_contours(self, mask, img, leaf_image, leafArea):
        # Find the contours of the segmented disease spots
        leaf_image = cv.cvtColor(leaf_image, cv.COLOR_BGR2RGB)
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        classifier = Classify(contours, self.leafArea,
                              self.image_path, self.model_path)
        classifier.classifyROI()
