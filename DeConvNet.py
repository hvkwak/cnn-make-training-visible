'''
    Codes mostly courtesy of: 
    https://github.com/saketd403/Visualizing-and-Understanding-Convolutional-neural-networks
'''
# Import necessary libraries
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU Debug Log Output
import numpy as np
import sys
import datetime
import cv2 as cv
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Flatten, Activation, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
# from tensorflow.keras.activations import *
# from tensorflow.keras.applications import imagenet_utils
import tensorflow.keras.backend as K
import numpy
import math
# import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import imagenet_utils

tf.compat.v1.disable_eager_execution()

class MyCallback(tf.keras.callbacks.Callback):

    def __init__(self, model):
        self.model = model
        self.directory_ = '/home/hyobin/Documents/test'
        self.layer_bis = int(input("layer bis? ex: Input-Conv-Conv-MaxP = 3"))
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
        
        test_images, _ = load_images(False)
        feature_maps_subset = [3, 7, 10]###
        results_mat = np.zeros((len(feature_maps_subset), len(test_images)))
        
        zeit = str(datetime.datetime.now())
        datum = zeit[:-16]+'_'
        minute = zeit[-15:-7]
        time_ = datum+minute
        
        
        # Full trained model results:
        if epoch == 0: ### epoch change this to 50.
            target_dir = self.directory_+'MNIST_full_trained_top5_'+str(self.model.name)+'_'+time_+'/'
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            
            os.chdir(target_dir)
            layer_dicts = [(layer.name, layer) for layer in self.model.layers]
            for layer_num in range(1, self.layer_bis):
                layer_name, _ = layer_dicts[layer_num]
                intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
                
                for i in range(len(test_images)): # for each images, take top 9 activations
                    intermediate_output = intermediate_layer_model.predict(test_images[i][np.newaxis, :])
                    for k in range(len(feature_maps_subset)):
                        results_mat[k, i] = np.mean(intermediate_output[0, :, :, feature_maps_subset[k]])
                
                
                for k in range(len(feature_maps_subset)):
                    image_nums = results_mat[k, :].argsort()[-5:][::-1] # take 5 values.
                    # print(image_nums)
                    # save top 5
                    # save top 5 deconv
                    feature_to_visualize = feature_maps_subset[k]
                    visualize_mode = 'all'
                    for img_num in image_nums: # top five save using for loop
                        img_array, filname_ = top_five_save(feature_to_visualize, layer_num, img_num, target_dir, test_images) # save top 5 pictures.
                        deconv = process_deconv(self.model, img_array, layer_name, feature_to_visualize, visualize_mode)
                        deconv_save2(deconv, target_dir, filname_)
                
                print("")
            
        else:
            print("")

        # deconv image at each epoch.
        if epoch == 0 or epoch == 1 or epoch == 4 or epoch == 9 or epoch == 19 or epoch == 49 :
            target_dir = self.directory_+'MNIST_deconv_'+str(self.model.name) + '_'+time_+'/'
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            os.chdir(target_dir)
            # load a random image
            img_array = load_an_image()
            
            # a list of dictionaries
            layer_dicts = [(layer.name, layer) for layer in self.model.layers] # convert keys of dictionary to list
            # print(layer_dicts)
            for layer_num in range(1, self.layer_bis): # jump over the input layer then visualize first five layers.
                layer_name, _ = layer_dicts[layer_num]
                feature_to_visualize = 2
                visualize_mode = 'all'   

                # Deconv
                print("process deconv: ", layer_name)
                deconv = process_deconv(self.model, img_array, layer_name, feature_to_visualize, visualize_mode)

                # save an random image deconvolutions at each layer
                deconv_save(deconv, layer_name, feature_to_visualize, epoch, layer_num, target_dir)
                '''
                A = np.min(deconv) - 0.00001
                deconv_img = (deconv - A)/np.max(deconv - A)
                plt.imshow(deconv_img)
                plt.show()
                '''
        else:
            print("pass the epoch: ", epoch+1)
            

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
        self.up_data = np.expand_dims(self.up_data, axis=0) ### here to decrease.
        # print(self.up_data.shape)
        return(self.up_data)

    def down(self, data, learning_phase = 0):
        # Backward pass
        self.down_data= self.down_func([data, learning_phase])
        self.down_data=np.squeeze(self.down_data,axis=0)
        self.down_data=numpy.expand_dims(self.down_data,axis=0)
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
        self.up_data = np.expand_dims(self.up_data,axis=0)
        # print(self.up_data.shape)
        return(self.up_data)

   
    def down(self, data, learning_phase = 0):
        
        self.down_data = self.down_func([data, learning_phase])
        self.down_data=np.squeeze(self.down_data,axis=0)
        self.down_data = np.expand_dims(self.down_data,axis=0)
        # print(self.down_data.shape)
        return(self.down_data)

class DInput(object):

    def __init__(self, layer):
        self.layer = layer
    
    # input and output of Inputl layer are the same
    def up(self, data, learning_phase = 0):
 
        self.up_data = data
        return(self.up_data)
    
    def down(self, data, learning_phase = 0):

        data = np.squeeze(data,axis=0)
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
        self.up_data=numpy.expand_dims(self.up_data,axis=0)
        # print(self.up_data.shape)
        return(self.up_data)
        
    def down(self, data, learning_phase = 0):
    
        self.down_data = self.down_func([data, learning_phase])
        self.down_data=np.squeeze(self.down_data,axis=0)
        self.down_data=numpy.expand_dims(self.down_data,axis=0)
        # print(self.down_data.shape)
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
        self.up_data = np.expand_dims(self.up_data,axis=0)
        # print(self.up_data.shape)
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
        # print(row_poolsize)
        # print(col_poolsize)
        out_shape[1] = math.floor(out_shape[1] / poolsize[0])
        out_shape[2] = math.floor(out_shape[2] / poolsize[1])
        # print(out_shape)
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

        # print('switch '+str(switch.shape))
        # print('input  '+str(input.shape))
        tile = np.ones((math.floor(switch.shape[1] / input.shape[1]), 
            math.floor( switch.shape[2] / input.shape[2])))
        # print('tile '+str(tile.shape))
        tile = np.expand_dims(tile, axis=3)
        input = np.squeeze(input, axis=0)
        out = np.kron(input, tile)
        # print('out '+str(out.shape))
        unpooled = out * switch
        unpooled = np.expand_dims(unpooled, axis=0)
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
    
    # print("stacked: ", deconv_layers)
    # Forward pass
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    output = deconv_layers[-1].up_data
    # print("output.shape: ", output.shape)
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
    # imports a test image
    (_, _), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    random_img_num = 33
    if K.image_data_format() == 'channels_first':
        # x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255
    x_test /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    print(x_test[33, :, :, :][np.newaxis, :], 'test sample shape')
    return(x_test[33, :, :, :][np.newaxis, :])

def load_images(train=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if train:
        return(x_train[0:200, :, :, :], y_train[0:200])
    else:
        return(x_test[0:100, :, :, :], y_test[0:100])

'''
def load_an_image(image_path = '/home/hkwak/Documents/newrpsdata/test/scissors/S_100.png'):
    # image_path = '/home/hkwak/Documents/rpsdata/test/paper/testpaper01-05.png'
    png = Image.open(image_path)
    png.load() # required for png.split()
    
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    # img = img.resize((300, 300),resample=Image.NEAREST)
    
    img = np.array(background)
    # img = cv.imread(image_path, cv.IMREAD_COLOR) # BGR image
    # img = cv.resize(img, (300, 300))
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.show()
    img = img[np.newaxis, :]
    # img = (img - np.mean(img))/255.0
    img = imagenet_utils.preprocess_input(x = img)/255.0
    return(img)


def load_images(train=True):
    if train:
        path1 = '/home/hkwak/Documents/newrpsdata/train/rock/'
        path2 = '/home/hkwak/Documents/newrpsdata/train/paper/'
        path3 = '/home/hkwak/Documents/newrpsdata/train/scissors/'
        r = "R_"
        p = "P_"
        s = "S_"
        imgs_per_class = 840 # 840
        images = [None]*(imgs_per_class*3)
        labels = [None]*(imgs_per_class*3)
        k = 0
        for i in range(1, imgs_per_class+1):
            # print(i)
            images[k*imgs_per_class + i-1] = load_an_image(path1+r+str(i)+'.png')[0, :, :, :]
            labels[k*imgs_per_class + i-1] = 0
        k = k + 1
        for i in range(1, imgs_per_class+1):
            # print(i)
            images[k*imgs_per_class + i-1] = load_an_image(path2+p+str(i)+'.png')[0, :, :, :]
            labels[k*imgs_per_class + i-1] = 1
        k = k + 1
        for i in range(1, imgs_per_class+1):
            # print(i)
            images[k*imgs_per_class + i-1] = load_an_image(path3+s+str(i)+'.png')[0, :, :, :]
            labels[k*imgs_per_class + i-1] = 2
        return(images, labels)
    else:
        path1 = '/home/hkwak/Documents/newrpsdata/test/rock/'
        path2 = '/home/hkwak/Documents/newrpsdata/test/paper/'
        path3 = '/home/hkwak/Documents/newrpsdata/test/scissors/'
        r = "R_"
        p = "P_"
        s = "S_"
        imgs_per_class = 124 # 840
        images = [None]*(imgs_per_class*3)
        labels = [None]*(imgs_per_class*3)
        k = 0
        # rock
        for i in range(1, imgs_per_class+1):
            # print(i)
            images[k*imgs_per_class + i-1] = load_an_image(path1+r+str(i)+'.png')[0, :, :, :]
            labels[k*imgs_per_class + i-1] = 0
        k = k+1
        for i in range(1, imgs_per_class+1):
            # print(i)
            images[k*imgs_per_class + i-1] = load_an_image(path2+p+str(i)+'.png')[0, :, :, :]
            labels[k*imgs_per_class + i-1] = 1
        k = k+1
        for i in range(1, imgs_per_class+1):
            # print(i)
            images[k*imgs_per_class + i-1] = load_an_image(path3+s+str(i)+'.png')[0, :, :, :]
            labels[k*imgs_per_class + i-1] = 2
        return(images, labels)
'''
    
def top_five_save(feature_map_num, layer_num, image_num, target_dir, test_images):
    sub_dir = 'Layer{}_FeatureMap{}/'.format(layer_num+1, feature_map_num)
    if not os.path.isdir(target_dir+sub_dir):
        os.makedirs(target_dir+sub_dir)
    os.chdir(target_dir+sub_dir)
    
    img = test_images[image_num] 

    filename_ = 'image{}.jpg'.format(image_num)
    cv.imwrite(filename_, (img*255).astype(np.uint8))
    return(img[np.newaxis, :], filename_)

''' 
def top_five_save(feature_map_num, layer_num, image_num, diese):
    path1 = '/home/hkwak/Documents/newrpsdata/test/rock/'
    path2 = '/home/hkwak/Documents/newrpsdata/test/paper/'
    path3 = '/home/hkwak/Documents/newrpsdata/test/scissors/'
    r = "R_"
    p = "P_"
    s = "S_"
    # directory_ = '/home/hkwak/Documents/Workspace/cnn-make-training-visible/saved_top_activations/'

    # print("num:"+str(image_num)+" div:"+str(image_num//31)+" mod:"+str(image_num%31))
    # if not os.path.isdir(directory_+diese):
    #     os.makedirs(directory_+diese)
    if image_num <=123: # rock
        image_path = path1+r+str(image_num+1)+'.png'
        
        png = Image.open(image_path)
        png.load() # required for png.split()
        
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    
        # img = img.resize((300, 300),resample=Image.NEAREST)
        
        img = np.array(background)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        os.chdir(diese)
        filename_ = 'One_of_Top5_from_layer{}_FeatureMap{}_R_image{}.jpg'.format(layer_num+1, feature_map_num, image_num+1)
        cv.imwrite(filename_, img)
        img = load_an_image(image_path)
        return(img, filename_)
    elif 124 <= image_num and image_num <= 247:
        image_num = image_num - 124
        
        image_path = path2+p+str(image_num+1)+'.png'
        
        png = Image.open(image_path)
        png.load() # required for png.split()
        
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    
        # img = img.resize((300, 300),resample=Image.NEAREST)
        
        img = np.array(background)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        os.chdir(diese)
        filename_ = 'One_of_Top5_from_layer{}_FeatureMap{}_P_image{}.jpg'.format(layer_num+1, feature_map_num, image_num+1)
        cv.imwrite(filename_, img)
        img = load_an_image(image_path)
        return(img, filename_)
    else:
        image_num = image_num - 124 - 124
        image_path = path3+s+str(image_num+1)+'.png'
        
        png = Image.open(image_path)
        png.load() # required for png.split()
        
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    
        # img = img.resize((300, 300),resample=Image.NEAREST)
        
        img = np.array(background)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        os.chdir(diese)
        filename_ = 'One_of_Top5_from_layer{}_FeatureMap{}_S_image{}.jpg'.format(layer_num+1, feature_map_num, image_num+1)
        cv.imwrite(filename_, img)
        img = load_an_image(image_path)
        return(img, filename_)
    '''
    
def deconv_save(deconv, layer_name, feature_map_num, epoch, layer_num, target_dir):
    '''
    save_dir = os.path.join(os.getcwd(), 'saved_models'+'_'+str(model_name))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    '''
    # postprocess(bring it back to scale of 255) and save image
    # print(deconv.shape)
    # deconv = np.transpose(deconv, ())

    # clipping the image back to valid range for visualization : [0.0, 1.0]
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8) # prevents numeric problem to visualize it.. 
    # deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    image_path = target_dir
    # layer_name, which is 'layer_num'th layer from the architecture
    img.save(image_path + '\Layer{}_{}__FeatureMap{}_epoch{}.jpg'.format(layer_num+1, layer_name, feature_map_num, epoch+1))

def deconv_save2(deconv, target_dir, filename_):
    '''
    save_dir = os.path.join(os.getcwd(), 'saved_models'+'_'+str(model_name))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    '''
    # postprocess(bring it back to scale of 255) and save image
    # print(deconv.shape)
    # deconv = np.transpose(deconv, ())

    # clipping the image back to valid range for visualization : [0.0, 1.0]
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8) # prevents numeric problem to visualize it.. 
    # deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    image_path = target_dir + filename_[:-4] + '_Deconv.jpg'
    img.save(image_path)