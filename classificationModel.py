from CoffeeDiseaseTrain import ImagePreprocessor
import os
import numpy as np
from sklearn.model_selection import KFold
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.models import Model
from keras import optimizers,losses,activations,models
from keras.layers import Convolution2D,Dense,Input,Flatten,Dropout,MaxPooling1D,BatchNormalization,GlobalAveragePooling2D,Concatenate

### get the test,train,validation data and label

class TrainModel:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        #dataset_path ='/home/frexg/Artificial Intelligence Center/BROCOLE/'
 
        self.imagepreprocess = ImagePreprocessor(self.dataset_path)

        self.train_data = self.imagepreprocess.datasetToArray('train_data')

        self.test_data = self.imagepreprocess.datasetToArray('test_data')

        self.train_label = self.imagepreprocess.getTrainingLabel('train_label.csv')

        self.test_label = self.imagepreprocess.getTestingLabel('test_label.csv')

    def getModel(self):
        base_model = InceptionV3(include_top=False,input_shape=(299,299,3))
        base_model.trainable = False

        add_model = Sequential()

        add_model.add(base_model)
        add_model.add(GlobalAveragePooling2D())

        add_model.add(Dropout(0.5))
        add_model.add(Dense(4,activation='softmax'))

        model = add_model

        return model

     ### K-Fold Validation   

    def K_Fold_Validate(self,K = 5):
        num_folds = K
        acc_per_fold = []
        loss_per_fold = []

        kfold = KFold(n_splits=num_folds,shuffle=True)

        fold_no = 1

        for train,validate in kfold.split(self.train_data,self.train_data):
            model = self.getModel()

            #Compile Model
            model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop',
              metrics=['accuracy'])


            ## Generate Print
            print('-------------------------------------------------------------------')

            print(f'Training for fold {fold_no} ...')

            ## Fit data to model
            history = model.fit(self.train_data[train],self.train_label[train],
                      epochs=100,
                      batch_size=32
                      )

            ## Generalization metrics

            scores = model.evaluate(self.train_data[validate],self.train_label[validate], verbose=0)

            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]} \n {model.metrics_names[1]} of {scores[1] * 100}')

            acc_per_fold.append(scores[1] *100)
            loss_per_fold.append(scores[0])
            fold_no += 1

      # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
          print('------------------------------------------------------------------------')
          print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

        ## Save the model

        model.save('CoffeeNet_1.0')

if __name__ == "__main__":
  train = TrainModel('/home/frexg/Artificial Intelligence Center/BROCOLE/')
  train.K_Fold_Validate()
