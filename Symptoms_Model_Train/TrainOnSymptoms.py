""" 
    #### Model Training ###
    @author: Fraol Gelana
    @Institute:Artificial Intelligence Center
    @Date:December,2020
"""
import os
import numpy as np
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
##from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2

def getGeneretors(dataset_path):
    ## Create data generetor
    datagen = ImageDataGenerator(
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function = preprocess,
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


    ## get train generator
    train_generator = datagen.flow_from_directory(directory=os.path.join(dataset_path,'train'),
    class_mode="categorical",
    target_size=(224,224),
    batch_size=64)

    ## get validation generator
    valid_generator = datagen.flow_from_directory(directory=os.path.join(dataset_path,'val'),
    class_mode="categorical",
    target_size=(224,224),
    batch_size=64)

    ## get test generator
    test_generator = datagen.flow_from_directory(directory=os.path.join(dataset_path,'test'),
    class_mode="categorical",
    target_size=(224,224),
    batch_size=1)

    return train_generator,valid_generator,test_generator

def preprocess(img):
  img = img.astype(np.float32) * 1/255.0
  img = (img -0.5) * 2
  return img


def getModel():
    IMAGE_WIDTH = 224   ## 224 x 224 for MobileNetV2 
    IMAGE_HEIGHT = 224  ## 299 x 299 for InceptionV3
    
    ## Load the inception v3 model
    """base_model = InceptionV3(include_top=False,input_shape=(299,299,3))
    base_model.trainabel = False
    """
    ## Load the mobilenet_v2 model
    base_model = MobileNetV2(include_top=False,input_shape=(224,224,3))

    base_model.trainabel = True

    for layer in base_model.layers[:100]:
      layer.trainabel = False


    add_model = Sequential()

    add_model.add(base_model)
    add_model.add(layers.GlobalAveragePooling2D())
    add_model.add(layers.Dropout(0.2))

    add_model.add(layers.Dense(128,activation='relu'))
    add_model.add(layers.Dense(64,activation='relu'))
    add_model.add(layers.Dense(5,activation='softmax'))

    model = add_model

    return model

def trainModel(dataset_path):
    
    EPOCH_SIZE = 64;
    model = getModel()
    (train_generator,valid_generator,test_generator) = getGeneretors(dataset_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    
    # Initialize Optimizer
    opt = optimizers.RMSprop(lr = 0.0001)

    model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint("best_model", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', save_freq="epoch")

    history = model.fit_generator(generator=train_generator,
       steps_per_epoch=STEP_SIZE_TRAIN,
       validation_data=valid_generator,
       validation_steps=STEP_SIZE_VAL,
       epochs = EPOCH_SIZE,
       callbacks = [checkpoint])
    
    history_dict = history.history
    loss_values = history_dict['loss']
    validation_loss_values = history_dict['loss']

    epochs = range(1,EPOCH_SIZE + 1)

    plt.plot(epochs,loss_values,'bo',label="Training loss")
    plt.plot(epochs,validation_loss_values,'r',label="Validation loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    test_generator.reset()
    pred = model.predict_generator(generator=test_generator,steps=STEP_SIZE_TEST,verbose=1)

    model.save('.......Model name goes here .........')

if __name__ == '__main__':
    trainModel('............Place dataset path here.................')

