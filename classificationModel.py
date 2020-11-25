from Image_Preprocessor import ImagePreprocessor
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.models import Model
from keras import optimizers,losses,activations,models
from keras.layers import Convolution2D,Dense,Input,Flatten,Dropout,MaxPooling1D,BatchNormalization,GlobalAveragePooling2D,Concatenate

### get the test,train,validation data and label
dataset_path ='/home/frexg/Artificial Intelligence Center/BROCOLE/'
 
imagepreprocess = ImagePreprocessor(dataset_path)

train_data = imagepreprocess.datasetToArray('train_data')
print(train_data.shape)

test_data = imagepreprocess.datasetToArray('test_data')
print(train_data.shape)

train_label = imagepreprocess.getTrainingLabel('train_label.csv')
test_label = imagepreprocess.getTestingLabel('test_label.csv')


#Create a separate validation set

x_train = train_data[200:]
y_train = train_label[200:]

x_val = train_data[:200]
y_val = train_label[:200]

########### Transfer learing Model using InceptionV3 #############
base_model = InceptionV3(include_top=False,input_shape=(299,299,3))
base_model.trainable = False

add_model = Sequential()

add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())

add_model.add(Dropout(0.5))
add_model.add(Dense(5,activation='softmax'))

model = add_model

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train,y_train,
         epochs=10,
         batch_size=128,
         validation_data=(x_val,y_val))
