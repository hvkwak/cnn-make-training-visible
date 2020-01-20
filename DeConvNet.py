'''
    Codes mostly courtesy of: 
    https://github.com/saketd403/Visualizing-and-Understanding-Convolutional-neural-networks
'''
# Import necessary libraries
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU Debug Log Output
import numpy as np
import sys
import time
import cv2 as cv
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, InputLayer, Flatten, Activation, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import imagenet_utils
import tensorflow.keras.backend as K
import numpy
import math
import matplotlib.pyplot as plt
from PIL import Image

class MyCallback(tf.keras.callbacks.Callback):

    def __init__(self, model):
        self.model = model
    
        '''
        self.image_array = 
        self.layer_name = 
        self.feature_to_visualize = 
        self.visualize_mode = 

    def on_epoch_begin(self, epoch, logs=None): # store all weights before training epoch
        self.init_weights = []
        for curr_layer in self.model.layers[1:]:
            self.init_weights.append( curr_layer.weights[0].numpy() ) 
        '''
    def on_epoch_end(self, epoch, logs=None): # store all weights after training epoch

        # load a random image
        img_array = load_an_image()
        
        layer_names = [*dict([(layer.name, layer) for layer in self.model.layers])] # convert keys of dictionary to list
        for i in range(1, len(layer_names)): # jump over the input layer
            layer_name = layer_names[i]
            feature_to_visualize = 1 # we don't need this actually.
            visualize_mode = 'all'   # because we are visualizing all features

            # Deconv
            deconv = process_deconv(self.model, img_array, layer_name, feature_to_visualize, visualize_mode)
            
            # save an random image deconvolutions at each layer
            deconv_save(deconv, layer_name, feature_to_visualize, visualize_mode, epoch, self.model.name, i)
            '''
            A = np.min(deconv) - 0.00001
            deconv_img = (deconv - A)/np.max(deconv - A)
            plt.imshow(deconv_img)
            plt.show()
            '''

class DConvolution2D(object):
    
    def __init__(self, layer):

        self.layer = layer
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]
        
        filters = W.shape[3]
        up_row = W.shape[0]
        up_col = W.shape[1]
        input_img = keras.layers.Input(shape = layer.input_shape[1:])

        output=keras.layers.Conv2D(filters,(up_row,up_col),kernel_initializer=tf.constant_initializer(W),
                                   bias_initializer=tf.constant_initializer(b),padding='same')(input_img)
        self.up_func = K.function([input_img, K.learning_phase()], [output])
        # Deconv filter (exchange no of filters and depth of each filter)
        W = np.transpose(W, (0,1,3,2))
        # Reverse columns and rows
        W = W[::-1, ::-1,:,:]
        down_filters = W.shape[3]
        down_row = W.shape[0]
        down_col = W.shape[1]
        b = np.zeros(down_filters)
        input_d = keras.layers.Input(shape = layer.output_shape[1:])

        output=keras.layers.Conv2D(down_filters,(down_row,down_col),kernel_initializer=tf.constant_initializer(W),
                                   bias_initializer=tf.constant_initializer(b),padding='same')(input_d)
        self.down_func = K.function([input_d, K.learning_phase()], [output])

    def up(self, data, learning_phase = 0):
        # Forward pass
        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data, axis=0)
        # self.up_data = np.expand_dims(self.up_data, axis=0)
        # print(self.up_data.shape)
        return(self.up_data)

    def down(self, data, learning_phase = 0):
        # Backward pass
        self.down_data= self.down_func([data, learning_phase])
        self.down_data=np.squeeze(self.down_data,axis=0)
        # self.down_data=numpy.expand_dims(self.down_data,axis=0)
        # print(self.down_data.shape)
        return(self.down_data)

class DActivation(object):
    def __init__(self, layer, linear = False):
        
        self.layer = layer
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape = layer.output_shape)

        output = self.activation(input)
        # According to the original paper, 
        # In forward pass and backward pass, do the same activation(relu)
        # Up method
        self.up_func = K.function(
                [input, K.learning_phase()], [output])
        # Down method
        self.down_func = K.function(
                [input, K.learning_phase()], [output])

   
    def up(self, data, learning_phase = 0):
        
        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data,axis=0)
        # self.up_data = np.expand_dims(self.up_data,axis=0)
        print(self.up_data.shape)
        return(self.up_data)

   
    def down(self, data, learning_phase = 0):
        
        self.down_data = self.down_func([data, learning_phase])
        self.down_data=np.squeeze(self.down_data,axis=0)
        # self.down_data = np.expand_dims(self.down_data,axis=0)
        print(self.down_data.shape)
        return(self.down_data)

class DInput(object):

    def __init__(self, layer):
        self.layer = layer
    
    # input and output of Inputl layer are the same
    def up(self, data, learning_phase = 0):
 
        self.up_data = data
        return(self.up_data)
    
    def down(self, data, learning_phase = 0):

        # data = np.squeeze(data,axis=0)
        data=numpy.expand_dims(data,axis=0)
        self.down_data = data
        return(self.down_data)

class DDense(object):
  
    def __init__(self, layer):

        self.layer = layer
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]
        
        # Up method
        input = Input(shape = layer.input_shape[1:])
        output = keras.layers.Dense(layer.output_shape[1],
                 kernel_initializer=tf.constant_initializer(W), bias_initializer=tf.constant_initializer(b))(input)
        self.up_func = K.function([input, K.learning_phase()], [output])
        
        # Transpose W  for down method
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape = self.output_shape[1:])
        output = keras.layers.Dense(self.input_shape[1:],
                 kernel_initializer=tf.constant_initializer(W), bias_initializer=tf.constant_initializer(b))(input)
        self.down_func = K.function([input, K.learning_phase()], [output])
    

    def up(self, data, learning_phase = 0):
      
        self.up_data = self.up_func([data, learning_phase])
        self.up_data=np.squeeze(self.up_data,axis=0)
        # self.up_data=numpy.expand_dims(self.up_data,axis=0)
        print(self.up_data.shape)
        return(self.up_data)
        
    def down(self, data, learning_phase = 0):
    
        self.down_data = self.down_func([data, learning_phase])
        self.down_data=np.squeeze(self.down_data,axis=0)
        # self.down_data=numpy.expand_dims(self.down_data,axis=0)
        print(self.down_data.shape)
        return(self.down_data)

class DFlatten(object):

    def __init__(self, layer):
   
        self.layer = layer
        self.shape = layer.input_shape[1:]
        self.up_func = K.function(
                [layer.input, K.learning_phase()], [layer.output])

    # Flatten 2D input into 1D output
    def up(self, data, learning_phase = 0):

        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data, axis=0)
        # self.up_data = np.expand_dims(self.up_data,axis=0)
        print(self.up_data.shape)
        return(self.up_data)

    # Reshape 1D input into 2D output
    def down(self, data, learning_phase = 0):

        new_shape = [data.shape[0]] + list(self.shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return(self.down_data)

class DPooling(object):

    def __init__(self, layer):

        self.layer = layer
        self.poolsize = layer.pool_size
       
    
    def up(self, data, learning_phase = 0):

        [self.up_data, self.switch] = \
                self.__max_pooling_with_switch(data, self.poolsize)
        return(self.up_data)

    def down(self, data, learning_phase = 0):
      
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return(self.down_data)
    
    def __max_pooling_with_switch(self, input, poolsize):

        switch = np.zeros(input.shape)
        out_shape = list(input.shape)
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        print(row_poolsize)
        print(col_poolsize)
        out_shape[1] = math.floor(out_shape[1] / poolsize[0])
        out_shape[2] = math.floor(out_shape[2] / poolsize[1])
        print(out_shape)
        pooled = np.zeros(out_shape)
        
        for sample in range(input.shape[0]):
            for dim in range(input.shape[3]):
                for row in range(out_shape[1]):
                    for col in range(out_shape[2]):
                        patch = input[sample, 
                                row * row_poolsize : (row + 1) * row_poolsize,
                                col * col_poolsize : (col + 1) * col_poolsize,dim]
                        max_value = patch.max()
                        pooled[sample, row, col,dim] = max_value
                        max_col_index = patch.argmax(axis = 1)
                        max_cols = patch.max(axis = 1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample, 
                                row * row_poolsize + max_row, 
                                col * col_poolsize + max_col,
                              dim]  = 1
        return([pooled, switch])
    
    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):

        print('switch '+str(switch.shape))
        print('input  '+str(input.shape))
        tile = np.ones((math.floor(switch.shape[1] / input.shape[1]), 
            math.floor( switch.shape[2] / input.shape[2])))
        print('tile '+str(tile.shape))
        tile = np.expand_dims(tile, axis=2)
        input = np.squeeze(input, axis=0)
        out = np.kron(input, tile)
        print('out '+str(out.shape))
        unpooled = out * switch
        # unpooled = np.expand_dims(unpooled, axis=0)
        return(unpooled)

def process_deconv(model, data, layer_name, feature_to_visualize, visualize_mode):
    deconv_layers = []
    # Stack layers
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], Convolution2D):
            deconv_layers.append(DConvolution2D(model.layers[i]))
            deconv_layers.append(
                    DActivation(model.layers[i]))
        elif isinstance(model.layers[i], MaxPooling2D):
            deconv_layers.append(DPooling(model.layers[i]))
        elif isinstance(model.layers[i], Dense):
            deconv_layers.append(DDense(model.layers[i]))
            deconv_layers.append(
                    DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Activation):
            deconv_layers.append(DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Flatten):
            deconv_layers.append(DFlatten(model.layers[i]))
        elif isinstance(model.layers[i], InputLayer):
            deconv_layers.append(DInput(model.layers[i]))
        else:
            print('Cannot handle this type of layer')
            print(model.layers[i].get_config())
            sys.exit()
        if layer_name == model.layers[i].name:
            break

    # Forward pass
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    output = deconv_layers[-1].up_data
    print("output.shape: ", output.shape)
    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:,:, :, feature_to_visualize]
    if 'max' == visualize_mode:
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp
    elif 'all' != visualize_mode:
        print('Illegal visualize mode')
        sys.exit()
    output = np.zeros_like(output)
    if 2 == output.ndim:
        output[:, feature_to_visualize] = feature_map
    else:
        output[:,: , :, feature_to_visualize] = feature_map

    # Backward pass
    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    deconv = deconv_layers[0].down_data
    deconv = deconv.squeeze()
    return(deconv)


def load_an_image():
    image_path = '~/Documents/rpsdata/test/paper/testpaper01-05.png'
    # img = Image.open(image_path)
    # img = np.array(img)
    img = cv.imread(image_path, cv.IMREAD_COLOR) # BGR image
    img = cv.resize(img, (300, 300))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.show()
    img = img[np.newaxis, :]
    img = img.astype(np.float)
    img = imagenet_utils.preprocess_input(x = img, mode = 'caffe')/255.0
    return(img)

def deconv_save(deconv, layer_name, feature_to_visualize, visualize_mode, epoch, model_name, i):
    save_dir = os.path.join(os.getcwd(), 'saved_models'+'_'+str(model_name))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # postprocess(bring it back to scale of 255) and save image
    print(deconv.shape)
    #deconv = np.transpose(deconv, ())

    # clipping the image back to valid range for visualization : [0.0, 1.0]
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8) # prevents numeric problem to visualize it.. 
    # deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    image_path = '~/Documents/Workspace/cnn-make-training-visible/saved_images/'
    # layer_name, which is 'i'th layer of the architecture
    img.save(image_path + '\{}_Layer{}_{}_epoch{}.png'.format(model_name, i, layer_name, epoch+1))
