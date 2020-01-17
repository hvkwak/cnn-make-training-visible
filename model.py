'''
Some of the useful links: build a classifier using keras 
https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d#file-classifier_from_little_data_script_1-py-L49

Dataset is available at: 
https://www.kaggle.com/sanikamal/rock-paper-scissors-dataset

Please note that...:
### : Aufgabestellung - Name
##  : Comments
#   : disables the code
'''

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU Debug Log Output
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
# from tensorflow.keras.applications import vgg16, imagenet_utils
import math
import matplotlib.pyplot as plt
import PIL
import matplotlib.image
from PIL import Image
# tf.enable_eager_execution()
from DeConvNet import MyCallback, DConvolution2D, DActivation, DInput, DDense, DFlatten, DPooling, visualize

#################################################################
#                  Load Data and Preprocessing                  #
#################################################################
input_shape = (300,300,3)
num_classes = 3
train_data_dir = '/home/hyobin/Documents/WiSe1920/CVDL/dataset/rock-paper-scissors-dataset/rock-paper-scissors/train' # Change for run
validation_data_dir = '/home/hyobin/Documents/WiSe1920/CVDL/dataset/rock-paper-scissors-dataset/rock-paper-scissors/validation'
test_data_dir = '/home/hyobin/Documents/WiSe1920/CVDL/dataset/rock-paper-scissors-dataset/rock-paper-scissors/test'
nb_train_samples = 100 # 2520 images
nb_validation_samples = 100
epochs = 1
batch_size = 10
validation_step = nb_validation_samples // batch_size
steps_per_epoch = nb_train_samples // batch_size

# to save the model
print("current directory: ", os.getcwd())
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_trained_model_with model_1.h5'

if K.image_data_format() == 'channels_first':
    input_shape = (3, 300, 300)
else:
    input_shape = (300,300,3)


# data generators and rescale them to 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(300, 300),
        batch_size=batch_size,
        classes = ["rock", "paper", "scissors"],
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(300, 300),
        batch_size=batch_size,
        classes = ["rock", "paper", "scissors"],
        class_mode='categorical')

#################################################################
#                  Model and Fit                                #
#################################################################
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(300,300,3)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes, activation='softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model2.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_step,
    callbacks=[MyCallback(model2)]
    )

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model2.save(model_path)
print('Saved trained model at %s ' % model_path)


#################################
### Another alternative model ###
#################################
'''
model3 = Sequential()
# Step 1 - Convolution
model3.add(Conv2D(32, (3, 3), padding='same', input_shape = input_shape, activation = 'relu'))
model3.add(Conv2D(32, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5)) # antes era 0.25

# Adding a second convolutional layer
model3.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5)) # antes era 0.25

# Adding a third convolutional layer
model3.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.5)) # antes era 0.25

# Step 3 - Flattening
model3.add(Flatten())
# Step 4 - Full connection
model3.add(Dense(units = 512, activation = 'relu'))
model3.add(Dropout(0.5)) 
model3.add(Dense(units = num_classes, activation = 'softmax'))

# Compiling the CNN
model3.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
model3.summary()
'''