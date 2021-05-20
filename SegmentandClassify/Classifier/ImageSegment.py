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
    leafImage = None
    fileName = None
    start_time = None
    stop_time = None
    final_thresh = None
    bg_subtracted = None
    acc = None

    def __init__(self, image_path, model_path, filename):
        self.image_path = image_path  # load image path
        self.model_path = model_path  # load saved deep learning model path
        self.fileName = filename
        self.currentMeanArray = []      # Empty array to hold the calculated Mean,
        self.currentVarianceArray = []  # Variance and
        self.areaUnderArray = []        # area under the mean-variance line

        self.readImage(self.image_path)  # read image

    def readImage(self, image_path):

        leaf_image = cv.imread(image_path)
        self.leafImage = leaf_image

        plt.imshow(cv.cvtColor(leaf_image, cv.COLOR_BGR2RGB))
        plt.show()

        self.remove_background(leaf_image)

        # OpenCV reads all images in BGR color space by default

    def remove_background(self, leaf_image):

        # Gaussian blur image to remove noise
        blured = cv.GaussianBlur(leaf_image, (1, 1), 0)

        self.bg_subtracted = cv.cvtColor(
            blured, cv.COLOR_BGR2RGB)

        # Convert blured Image from BGR to HSV
        hsv_leaf = cv.cvtColor(blured, cv.COLOR_BGR2HSV)

        SV_channel = hsv_leaf.copy()

        SV_channel[:, :, 0] = np.zeros(
            (SV_channel.shape[0], SV_channel.shape[1]))  # Set the 'H' channel to Zero

        # SV_channel[:, :, 2] = np.zeros(
        #    (SV_channel.shape[0], SV_channel.shape[1]))
        # Create a binary mask from the SV Channel

        mask = cv.inRange(SV_channel, (0, 0, 80), (0, 90, 255))

        # Invert mask, White areas represent green components and black the background
        mask = cv.bitwise_not(mask)

        contours, heirarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        self.leafArea = w * h

        # perform bitwise_and between mask and hsv image

        # ret, mask = cv.threshold(
        #    hsv_leaf[:, :, 1], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        background_extracted = cv.bitwise_and(hsv_leaf, hsv_leaf, mask=mask)

        plt.imshow(cv.cvtColor(background_extracted, cv.COLOR_HSV2RGB))
        plt.show()

        self.color_segment(background_extracted)

    def color_segment(self, hsv_space, lowerB=(36, 0, 0), upperB=(65, 255, 255), count=2):
        # extracted in the HSV color space
        self.start_time = time.time()
        # create binary mask using the bounds
        mask = cv.inRange(hsv_space, lowerB, upperB)
        mask = cv.bitwise_not(mask)

        # bitwise_and mask and rgb image
        output_hsv = cv.bitwise_and(hsv_space, hsv_space, mask=mask)

        nonZeroIntentsity = output_hsv[:, :, 0].copy()

        # Extract intensity values between [3,65]
        # This step ensures the black(intensity 0 - 2) pixel values
        # from the mean calculation.
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity > 3]
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity < 255]

        l = self.findLowerBound(nonZeroIntentsity)
        # Update the lower bound value of the 'H' channel accordingly
        # (H,S,V)
        new_lowerB = (l, 0, 0)

        mask = cv.inRange(hsv_space, new_lowerB, upperB)
        mask = cv.bitwise_not(mask)

        plt.imshow(mask, cmap="gray")
        plt.show()

        # bitwise_and mask and rgb image

        o_hsv = cv.bitwise_and(output_hsv, output_hsv, mask=mask)
        self.thresh_mask(o_hsv)

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

    def thresh_mask(self, out):
        mask = out[:, :, 2]
        # Calculate the otsu threshold
        ret, thresh = cv.threshold(
            mask, 0, 55, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Morphological close operation
        kernel = np.ones((3, 3))

        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        #thresh = cv.bitwise_not(thresh)

        plt.imshow(thresh, cmap="gray")
        plt.show()

        self.final_thresh = thresh

        self.stop_time = time.time() - self.start_time
        # print(time_taken)

       # self.acc = Evaluator(thresh, self.fileName,
        #                     self.stop_time).getAccuracy()

        self.find_contours(thresh, out)

    def getThresh(self):
        return (self.final_thresh, self.bg_subtracted, self.stop_time, self.acc)

    def find_contours(self, mask, img):
        # Find the contours of the segmented disease spots
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        classifier = Classify(contours, self.leafArea,
                              self.image_path, self.model_path)
        classifier.classifyROI()
