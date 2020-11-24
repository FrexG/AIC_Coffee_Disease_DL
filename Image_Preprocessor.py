### Import dependencies and libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import os

class ImagePreprocessor:
    def __init__(self,path):
        self.dataset_path = path

    def datasetToArray(self,testSetFolderName):
        folderName = testSetFolderName
        ## Change the working direcory    
        os.chdir(self.dataset_path)

        data_path = f'{self.dataset_path}{folderName}'

        if os.path.exists(data_path):
            for dirpath,dirnames,filenames in os.walk(data_path):
                self.images = filenames

            images_Set = [self.imageToArray(i,data_path) for i in self.images]
            images_Set = np.array(images_Set)

        else:
            raise FileNotFoundError(f"Path {data_path} not found!!")
        return images_Set    


    def imageToArray(self,imageName,path):
        image = load_img(f'{path}/{imageName}',target_size = (299,299))
        ImgToArray = img_to_array(image)
        print(f'processing .... [{imageName}]')
        return ImgToArray        
    
    def getTrainingLabel(self,fileName):
        field = ['miner','rust','phoma','cercospora','spider_mite']
        
        os.chdir(self.dataset_path)
        if os.path.isfile(fileName):
            df = pd.read_csv(fileName,usecols=field)
            print(df)
            train_label = np.array(df)
            return train_label
        else:
            raise FileNotFoundError(f'file {fileName} not found!!')

    def getTestingLabel(self,fileName):
        field = ['miner','rust','phoma','cercospora','spider_mite']
        
        os.chdir(self.dataset_path)
        if os.path.isfile(fileName):
            df = pd.read_csv(fileName,usecols=field)
            test_label = np.array(df)
            return test_label  
        else:
            raise FileNotFoundError(f'file {fileName} not found!!')
