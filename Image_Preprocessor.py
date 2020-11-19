### Import dependencies and libraries
from keras.applications.inception_v3 import InceptionV#
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os

class ImagePreprocessor:
    def __init__(self,path):
        self.dataset_path = path

    def testsetToArray(self,testSetFolderName):
        folderName = testSetFolderName
        ## Change the working direcory    
        os.chdir(self.dataset_path)

        test_data_path = f'{self.dataset_path}{folderName}'

        if os.path.exists(self.test_data_path):
            for dirpath,dirnames,filenames in os.walk(test_data_path):
                self.test_images = filenames

            test_images_Set = [self.imageToArray(i,test_data_path) for i in self.test_images]
            test_images_Set = np.array(test_images_Set)

        else:
            raise FileNotFoundError(f"Path {test_data_path} not found!!")
        return test_images_Set    

    def trainsetToArray(self,trainSetFolderName):
        folderName = trainSetFolderName
        ## Change the working direcory    
        os.chdir(self.dataset_path)

        train_data_path = f'{self.dataset_path}{folderName}'

        if os.path.exists(train_data_path):
            for dirpath,dirnames,filenames in os.walk(train_data_path):
                self.train_images = filenames

            train_images_Set = [self.imageToArray(i,train_data_path) for i in self.train_images]
            train_images_Set = np.array(train_images_Set)

        else:
            raise FileNotFoundError(f"Path {self.test_data_path} not found!!")
        return train_images_Set  

    def imageToArray(self,imageName,path):
        image = load_img(f'{path}/{imageName}',target_size = (299,299))
        ImgToArray = img_to_array(image)
        print(f'processing .... [{imageName}]')
        return ImgToArray                  
