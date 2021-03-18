""" 
    #### Classification Class ###
    @author: Fraol Gelana
    @Institute:Artificial Intelligence Center
    @Date:January,2021
"""

import matplotlib.pyplot as plt
import cv2 as cv
from keras.models import load_model
import numpy as np


class Classify:
    contours = None
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    def __init__(self, contours, leafArea, image_path, model_path):
        # Initialize parameters
        self.imagePath = image_path
        self.modelPath = model_path
        self.contours = contours
        self.leafArea = leafArea
        print("Here!!")

    def openImage(self):
        # Open original image
        leaf_image = cv.imread(self.imagePath)
        leaf_image = cv.cvtColor(leaf_image, cv.COLOR_BGR2RGB)
        return leaf_image

    def loadModel(self):
        # load trained model
        model = load_model(self.modelPath)
        return model

    def classifyROI(self):

        leaf_image = self.openImage()
        loaded_model = self.loadModel()
        roi_values = []
        # Classify each object inside specified rectangular area
        # specified by the contour (x,y,width,height) of the contour
        for cnt in self.contours:
            x, y, w, h = cv.boundingRect(cnt)
            if (w * h) >= self.leafArea*0.001:

                # if x + 400 <= img.shape[0] and y + 400 <= img.shape[1]:
                input_image = leaf_image[y:y+h, x:x+w]
                # else:
                #    input_image = leaf_image[y:y+h, x:x+w]

                input_image = cv.resize(
                    input_image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
                input_image = input_image.reshape(-1,
                                                  self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)
                input_image = np.array(input_image)
                input_image = ((input_image * 1./255) - 0.5) * 2

                prediction = loaded_model.predict_classes(input_image)

                roi_values.append({'prediction': prediction[0],
                                   'x_cord': [x, x + w],
                                   'y_cord': [y, y + h]})

        diseases = ["CERCOSPORA", "HEALTHY", "MINER", "PHOMA", "RUST"]
        font_size = 0.9 if leaf_image.shape[0] <= 2048 else 5

        # Draw bounding rectangel and predicted label
        for val in roi_values:
            if val['prediction'] != 1:
                r = cv.rectangle(leaf_image, (val['x_cord'][0], val['y_cord'][0]),
                                 (val['x_cord'][1], val['y_cord'][1]), (5, 255, 10), 2)
                cv.putText(r, diseases[val['prediction']],
                           (val['x_cord'][0], val['y_cord'][0] - 5), cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
            plt.imshow(leaf_image)
        plt.show()
