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
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
# import matplotlib.image
# import os
# from PIL import Image
# tf.enable_eager_execution()

#################################################################
#                  Load Data and Preprocessing                  #
#################################################################

## Some parameters are set randomly. After the model is complete, 
## will change it back to normal.

### Get ready to load data and preprocess - Hyobin
input_shape = (300,300,3)
num_classes = 3
train_data_dir = '/home/hyobin/Documents/WiSe1920/CVDL/dataset/rock-paper-scissors-dataset/rock-paper-scissors/train' # Change for run
validation_data_dir = '/home/hyobin/Documents/WiSe1920/CVDL/dataset/rock-paper-scissors-dataset/rock-paper-scissors/validation'
test_data_dir = '/home/hyobin/Documents/WiSe1920/CVDL/dataset/rock-paper-scissors-dataset/rock-paper-scissors/test'
nb_train_samples = 100
nb_validation_samples = 10
epochs = 1
batch_size = 10

if K.image_data_format() == 'channels_first':
    input_shape = (3, 300, 300)
else:
    input_shape = (300,300,3)

### Preprocess the data properly - Hyobin
# normalize images.
datagen = ImageDataGenerator(
    # normalize images, featurewise.
    featurewise_center=True,
    featurewise_std_normalization=True)
# load data: already done for Simon :D
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
### convert class vectors to binary class matrices - Hyobin
# class_mode = 'categorical' will take care of this.


print("hi")

###########################################################################
# Implement the architecture from original paper(Zeiler and Fergus, 2014) #
###########################################################################

model = Sequential()
### Implement Layer 1 - Arslan
model.add(Conv2D(filters=96, kernel_size=(7,7), strides=2, 
                 activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(3,3), strides=2))

## Implement these two if needed:
### The contrast normaliztion for layer 1(should be implemented here) - Arslan
raise NotImplementedError('Implement me')

### model.add(normalization.BatchNormalization()) - Arslan
raise NotImplementedError('Implement me')

### Implement Layer 2 - Arslan
model.add(Conv2D(filters=256, kernel_size=(7,7), strides=2, 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=2))



### Implement Layer 3 - Aya
raise NotImplementedError('Implement me')
### Implement Layer 4 - Aya
raise NotImplementedError('Implement me')
### Implement Layer 5 - Aya
raise NotImplementedError('Implement me')
### Implement Layer 6 - Aya
raise NotImplementedError('Implement me')



### Implement Layer 7 and summary() - Dimitri
model.add(Dense(num_classes, activation='softmax'))
model.summary()
### Implement model.compile() - Dimitri
raise NotImplementedError('Implement me')
### Implement train_datagen = ImageDataGenerator() - Dimitri
raise NotImplementedError('Implement me')





## Implement rest of ImageDataGenerators and fit_generator
### Implement test_datagen = ImageDataGenerator() - Simon
raise NotImplementedError('Implement me')
### Implement train_generator = train_datagen.flow_from_directory() - Simon
raise NotImplementedError('Implement me')
### Implement test_generator = test_datagen.flow_from_directory() - Simon
raise NotImplementedError('Implement me')
### Implement model.fit_generator() - Simon
raise NotImplementedError('Implement me')











# We will not use the alternatives, won't we? Let us forget about these:
# Alternative models suggested by Arslan
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(200,300,3)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss=categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model2.summary()

#################################
### Another alternative model ###
#################################

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
