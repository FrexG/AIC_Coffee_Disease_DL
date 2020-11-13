import os
import cv2 as cv
import pandas as pd
import numpy as np

"""
@Fraol Gelana
Version: 0.1.0
November 13 2020
"""
class DataPreprocessor:
    def __init__(self,path):
        ## Initialize the folder containing the dataset
        self.dataset_path = path

    def train_test_split(self):
        ## Change the working directory to 'path'
        if os.path.exists(self.dataset_path):
            
            os.chdir(self.dataset_path)

            ### Read the 'dataset.csv' label file
            ### and convert it into a numpy array
            if os.path.exists(f'{self.dataset_path}dataset.csv'):
                ## Read the CSV file and convert it into a pandas dataframe
                self.dataset_labelCSV = pd.read_csv('dataset.csv')

                ## Convert the dataframe into a numpy array
                self.dataset_labels = np.array(self.dataset_labelCSV)

                ### Split the dataset labels into 'train' and 'test' labels
                ### 70% train labels and 30% test labels

                sizeOfdataset_labels = len(self.dataset_labels)
                dataBatch = int(sizeOfdataset_labels * 0.7) ## this will give the size of train labels

                self.train_label = np.array(self.dataset_labels[:dataBatch])
                self.test_label = np.array(self.dataset_labels[dataBatch:])

            else:
                raise FileNotFoundError("File Doesn't Exist!!")
                return
            ### Split the dataset images into 'train' and 'test' set
            ### First create two directories names 'train_data' and 'test_data'

            os.mkdir('test_data')
            os.mkdir('train_data')


            for label in self.train_label:

                if os.path.exists(f'{self.dataset_path}{label[0]}.jpg'):
                    # If the image exists, then copy it to the 'train_data' folder
                    os.rename(f'{self.dataset_path}{label[0]}.jpg',f'{self.dataset_path}train_data/{label[0]}.jpg')
                else:
                    print(f"{label[0]}.jpg doesn't exist")
                    continue

            for label in self.test_label:

                if os.path.exists(f'{self.dataset_path}{label[0]}.jpg'):
                    # If the image exists, then copy it to the 'test_data' folder
                    os.rename(f'{self.dataset_path}{label[0]}.jpg',f'{self.dataset_path}test_data/{label[0]}.jpg')
                else:
                    print(f"{label[0]}.jpg doesn't exist")
                    continue
        else:
            raise FileNotFoundError(f"The path '{self.dataset_path}' doesn't exist!!")

        return (self.train_label,self.test_label)


if __name__ == "__main__":
    datasetPath = '/media/frextm/FrexData2/BROCOLE/' ## Absolute Path of the dataset folder 
    preproccessing = DataPreprocessor(datasetPath)   ## Create instance of class DataPreprocessor(path)   
    train_label,test_label = preproccessing.train_test_split() ## Unpack the tuple into two variables
    train_label.shape
    test_label.shape
