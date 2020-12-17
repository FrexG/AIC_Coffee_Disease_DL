### Import dependencies and libraries
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import cv2 as cv
import os

class ImagePreprocessor:
    def __init__(self,path):
        self.dataset_path = path

    def datasetToArray(self,testSetFolderName):
        folderName = testSetFolderName
        ## Change the working direcory    
        os.chdir(self.dataset_path)

        test_data_path = f'{self.dataset_path}{folderName}'

        if os.path.exists(test_data_path):
            for dirpath,dirnames,filenames in os.walk(test_data_path):
                self.images = filenames

            test_images_Set = [self.imageToArray(i,test_data_path) for i in self.images]
            test_images_Set = np.array(test_images_Set)

        else:
            raise FileNotFoundError(f"Path {test_data_path} not found!!")
        return test_images_Set    


    def imageToArray(self,imageName,path):
        #read image
        image = cv.imread(f'{path}/{imageName}')
        
        #convert image to grayscale
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        
        #Threshold the image and convert it binary
        
        thresholdValue = 185
        
        ret,thresh = cv.threshold(gray,thresholdValue,255,1)
        
        # Find the contours of the image
        
        contours,hierarchy =cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        # Find the largest contour by areafrom the list of contours
        
        largestContour = max(contours,key=cv.contourArea)
        
        #get the bounding rectangle of the largest contour
        
        x,y,w,h = cv.boundingRect(largestContour)
        
        # Crop the image around the bounding rectagle
        
        image = image[y:y+h,x:x+w]
        
        # Resize the cropped image to (299,299)
        dim = (299,299)
        image = cv.resize(image,dim)
        
        #Convert the image into array
             
        #image = load_img(f'{path}/{imageName}',target_size = (299,299))
        ImgToArray = img_to_array(image)
        
        # Normalize the array, divide by 255
       
        ImgToArray = ImgToArray * (1./255)
        
        print(f'processing .... [{imageName}]')
        return ImgToArray        
    
    def getTrainingLabel(self,fileName):
        field = ["rust","phoma","cercospora","healthy"]
        
        os.chdir(self.dataset_path)
        if os.path.isfile(fileName):
            df = pd.read_csv(fileName,usecols=field)
            print(df)
            train_label = np.array(df)
            return train_label
        else:
            raise FileNotFoundError(f'file {fileName} not found!!')

    def getTestingLabel(self,fileName):
        field = ["rust","phoma","cercospora","healthy"]
        
        os.chdir(self.dataset_path)
        if os.path.isfile(fileName):
            df = pd.read_csv(fileName,usecols=field)
            test_label = np.array(df)
            return test_label  
        else:
            raise FileNotFoundError(f'file {fileName} not found!!')
