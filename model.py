from __future__ import print_function

# import utils
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K
from DeConvNet import MyCallback, DConvolution2D, DActivation, DInput, DDense, DFlatten, DPooling, load_an_image, deconv_save, load_images
import numpy as np
import math
import matplotlib.pyplot as plt
import PIL
import matplotlib.image
from PIL import Image


#################################################################
#                  Load Data and Preprocessing                  #
#################################################################
input_shape = (300,300,3)
num_classes = 3

# change these values.
epochs = 50 # 50
batch_size = 128 # 128
# imgs_per_class = 10 # 840
# imgs_per_class = 124 # 
# imgs_per_class = 124 # 
# epoch change to 50.
# subset [3] war [3, 7, 10]

'''
train_datagen = ImageDataGenerator(        
        # featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,
        rescale = 1./255)  # divide each input by its std)
test_datagen = ImageDataGenerator(
        # featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,
        rescale = 1./255)  # divide each input by its std

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
'''
print("Load images....")
train_x, train_y = load_images(True)
test_x, test_y = load_images(False)


train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
print("wait here")
# print('train_x shape:', train_x.shape)
# print(x_train.shape[0], 'train samples')


# print('test_x shape:', test_x.shape)
# print(test_x.shape[0], 'test samples')
train_y = keras.utils.to_categorical(train_y)
test_y = keras.utils.to_categorical(test_y)

#################################################################
#                  Models and Fit                               #
#################################################################
whichmodel = input("model2 or model3")

if whichmodel == "model2":
    # Model 2 - Arslan's alternative suggestion
    inputs0 = Input(shape=(300, 300, 3,))
    x = Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding = 'same',
                     input_shape=(300,300,3))(inputs0)
    x = MaxPooling2D(pool_size=(2, 2))(x)       
    x = Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding = 'same',
                     input_shape=(150, 150,3))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding = 'same',
                     input_shape=(75, 75,3))(x)
    x = MaxPooling2D(pool_size=(5, 5))(x)                                  
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model_func = Model(inputs=inputs0, outputs=predictions, name = 'model2')
    
    model_func.summary()
    model_func.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    
    model_func.fit(
        train_x, train_y,
        batch_size = batch_size,
        epochs=epochs,
        shuffle = True,
        verbose=1,
        validation_data=(test_x, test_y),
        callbacks=[MyCallback(model_func)]
        )
    print("current directory: ", os.getcwd())
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_trained_model_with_model_2.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model_func.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    # load 2 images
    normal_img = load_an_image() # class: paper
    img_blacked = load_an_image()[0, :, :, :] 
    img_blacked[100:200, 100:200, :] = 0
    print("model_func prediction: normal, blacked")
    print("model_func.predict(normal_img): ", model_func.predict(normal_img))
    print("model_func.predict(img_blacekd): ", model_func.predict(img_blacked[np.newaxis, :]))
    model_func.evaluate(test_x, test_y)
else: # model3
    # Model 3 - Arslan's another alternative suggestion
    inputs1 = Input(shape=(300, 300, 3,))
    # Step 1 - Convolution
    x1 = Conv2D(32, (3, 3), padding='same', input_shape = (300, 300, 3), activation = 'relu')(inputs1)
    x1 = Conv2D(32, (3, 3), padding = 'same', input_shape = (300, 300, 3), activation='relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    
    # Adding a second convolutional layer
    x1 = Conv2D(64, (3, 3), padding='same', input_shape = (150, 150, 3), activation = 'relu')(x1)
    x1 = Conv2D(64, (3, 3), padding = 'same', input_shape = (150, 150, 3), activation='relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    
    # Adding a third convolutional layer
    x1 = Conv2D(64, (3, 3), padding ='same', input_shape = (75, 75, 3), activation = 'relu')(x1)
    x1 = Conv2D(64, (3, 3), padding = 'same', input_shape = (75, 75, 3), activation='relu')(x1)
    x1 = MaxPooling2D(pool_size=(5, 5))(x1)
    # x1 = Dropout(0.5)(x1)
    
    # Step 3 - Flattening
    x1 = Flatten()(x1)
    
    # Step 4 - Full connection
    x1 = Dense(32, activation = 'relu')(x1)
    # x1 = Dropout(0.5)(x1)
    predictions1 = Dense(units = num_classes, activation = 'softmax')(x1)
    model_func1 = Model(inputs=inputs1, outputs=predictions1, name='model3')
    
    # Compiling the CNN
    model_func1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    model_func1.summary()
    
    model_func1.fit(
        train_x, train_y,
        batch_size = batch_size,
        epochs=epochs,
        shuffle = True,
        verbose=1,
        validation_data=(test_x, test_y),
        callbacks=[MyCallback(model_func1)]
        )
    print("current directory: ", os.getcwd())
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name1 = 'keras_trained_model_with_model_3.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path1 = os.path.join(save_dir, model_name1)
    model_func1.save(model_path1)
    print('Saved trained model at %s ' % model_path1)
    # load 2 images
    normal_img = load_an_image() # class: paper
    img_blacked = load_an_image()[0, :, :, :] 
    img_blacked[100:200, 100:200, :] = 0
    print("model_func1 prediction: normal, blacked")
    print("model_func1.predict(normal_img): ", model_func1.predict(normal_img))
    print("model_func1.predict(img_blacekd): ", model_func1.predict(img_blacked[np.newaxis, :]))
    model_func1.evaluate(test_x, test_y)


################################################################
####                Save Models                             ####
################################################################


################################################################
####             Occluded image prediction                  ####
################################################################












'''
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
'''


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