import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
from Classifier.EvaluateSegment import Evaluator


class YCGCR:
    start_time = None
    stop_time = None
    fileName = None
    finalThresh = None
    acc = None

    def __init__(self, *filename, **image):
        self.fileName = filename[0]
        image_path = image["image"]
        bgr_image = cv.imread(image_path)
        self.remove_background(bgr_image)
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

        # perform bitwise_and between mask and hsv image

        # ret, mask = cv.threshold(
        #    hsv_leaf[:, :, 1], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        background_extracted = cv.bitwise_and(hsv_leaf, hsv_leaf, mask=mask)

        self.segment_diseased(background_extracted)

    def segment_diseased(self, bg_extracted_hsv):
        self.start_time = time.time()
        bg_extracted_rgb = cv.cvtColor(bg_extracted_hsv, cv.COLOR_HSV2RGB)

        r = bg_extracted_rgb[:, :, 0]
        g = bg_extracted_rgb[:, :, 1]
        b = bg_extracted_rgb[:, :, 2]

        #ycbcr_image = cv.cvtColor(bg_extracted_rgb, cv.COLOR_RGB2YCrCb)
        #ycbcr_image[:, :, 0] = np.zeros_like(ycbcr_image[:, :, 0])

        cr = 128 + (112 * r - 93.786 * g - 18.214 * b) / 256
        #cb = ycbcr_image[:, :, 2]
        # find the cg component
        #GY = -0.299 * b + 0.413*g - 0.114 * r

        cg = 128 + (-81.085 * r + 112 * g - 30.915 * b) / 256
        # print(cg.shape)

        diff = cr - cg

        g1, g2 = self.findThreshold(diff)

        self.stop_time = time.time() - self.start_time

        self.finalThresh = g2

        self.acc = Evaluator(g2, self.fileName, self.stop_time).getAccuracy()

        #ycbcr_image[:, :, 2] = np.zeros_like(ycbcr_image[:, :, 0])
        #lowerbound = (0, 76, 120)
        #upperbound = (0, 124, 176)

    def findThreshold(self, diff, T=0, count=0):
        T_init = T
        c = count
        diff_matrix = diff
        g1 = diff_matrix <= T

        g2 = diff_matrix > T

        m1 = np.mean(g1)
        m2 = np.mean(g2)

        Thresh = (m1 + m2) / 2
        deltaT = Thresh - T_init
        c = c + 1

        if c == 100 or deltaT < 0.001:
            return
        self.findThreshold(diff_matrix, T=Thresh, count=c)
        return (g1, g2)

    def getThresh(self):
        return (self.finalThresh, self.stop_time, self.acc)
