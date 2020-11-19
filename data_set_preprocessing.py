import os
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
                
                self.train_label = []
                self.test_label = []
                
                for i in range(0,1251):
                    self.train_label.append(self.dataset_labels[i])
    
                for i in range(1500,2500):
                    self.train_label.append(self.dataset_labels[i])
    

                for i in range(1251,1500):
                    self.test_label.append(self.dataset_labels[i])
    
                for i in range(2500,2965):
                    self.test_label.append(self.dataset_labels[i])
                
                self.train_label = np.array(self.train_label)
                self.test_label = np.array(self.test_label)
                
                ## Create a test_label.csv and train_label.csv
                
                self.train_label_DF = pd.DataFrame(data=self.train_label,columns=["id","miner",
                                                                                 "rust","phoma","cercospora",
                                                                                 "spider_mite"])
                
                self.test_label_DF = pd.DataFrame(data=self.test_label,columns=["id","miner",
                                                                                 "rust","phoma","cercospora",
                                                                                 "spider_mite"])
                self.test_label_DF.to_csv("test_label.csv",index=False)
                self.train_label_DF.to_csv("train_label.csv",index=False)


            else:
                raise FileNotFoundError("File Doesn't Exist!!")
                return
            ### Split the dataset images into 'train' and 'test' set
            ### First create two directories names 'train_data' and 'test_data'

            os.mkdir('test_data')
            os.mkdir('train_data')


            for label in self.train_label:

                if os.path.exists(f'{self.dataset_path}{int(label[0])}.jpg'):
                    # If the image exists, then copy it to the 'train_data' folder
                    os.rename(f'{self.dataset_path}{int(label[0])}.jpg',f'{self.dataset_path}train_data/{int(label[0])}.jpg')
                else:
                    print(f"{int(label[0])}.jpg doesn't exist")
                    continue

            for label in self.test_label:

                if os.path.exists(f'{self.dataset_path}{int(label[0])}.jpg'):
                    # If the image exists, then copy it to the 'test_data' folder
                    os.rename(f'{self.dataset_path}{int(label[0])}.jpg',f'{self.dataset_path}test_data/{int(label[0])}.jpg')
                else:
                    print(f"{label[0]}.jpg doesn't exist")
                    continue
        else:
            raise FileNotFoundError(f"The path '{self.dataset_path}' doesn't exist!!")

        return None

"""
if __name__ == "__main__":
    datasetPath = '/media/frextm/FrexData2/BROCOLE/' ## Absolute Path of the dataset folder 
    preproccessing = DataPreprocessor(datasetPath)   ## Create instance of class DataPreprocessor(path)   
    preproccessing.train_test_split() ## Unpack the tuple into two variables
""""   
