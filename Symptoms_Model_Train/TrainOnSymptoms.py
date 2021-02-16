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
from keras.models import Model, Sequential
from keras import layers
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3


def getGeneretors(dataset_path):
    # Create data generetor
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # get train generator
    train_generator = datagen.flow_from_directory(directory=os.path.join(dataset_path, 'train'),
                                                  class_mode="categorical",
                                                  target_size=(299, 299),
                                                  batch_size=32)

    # get validation generator
    valid_generator = datagen.flow_from_directory(directory=os.path.join(dataset_path, 'val'),
                                                  class_mode="categorical",
                                                  target_size=(299, 299),
                                                  batch_size=32)

    # get test generator
    test_generator = datagen.flow_from_directory(directory=os.path.join(dataset_path, 'test'),
                                                 class_mode="categorical",
                                                 target_size=(299, 299),
                                                 batch_size=1)

    return train_generator, valid_generator, test_generator


def getModel():
    # Load the inception v3 model
    base_model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    base_model.trainabel = False

    add_model = Sequential()

    add_model.add(base_model)
    add_model.add(layers.GlobalAveragePooling2D())

    add_model.add(layers.Dropout(0.2))
    add_model.add(layers.Dense(5, activation='softmax'))

    model = add_model

    return model


def trainModel(dataset_path):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = getModel()
    (train_generator, valid_generator, test_generator) = getGeneretors(dataset_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VAL,
                                  epochs=48)

    history_dict = history.history
    loss_values = history_dict['loss']
    validation_loss_values = history_dict['val_loss']

    epochs = range(1, 49)

    plt.plot(epochs, loss_values, 'bo', label="Training loss")
    plt.plot(epochs, validation_loss_values, 'r', label="Validation loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    test_generator.reset()
    pred = model.predict_generator(
        generator=test_generator, steps=STEP_SIZE_TEST, verbose=1)

    model.save('CoffeeNet_V2')


if __name__ == '__main__':
    trainModel('...place your dataset path here!!')
