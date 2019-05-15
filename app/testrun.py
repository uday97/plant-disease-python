from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import regularizers, optimizers
#from keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, Flatten, Dropout
from keras.layers import Dropout
#from keras.models import Model
import numpy as np
from keras.utils import np_utils
import os
import keras
#import pandas as pd
import time
from scipy.misc import imread, imresize
    
    
    
   
def le_net(drop):
    model = Sequential()
    # first set of CONV => RELU => POOL
    model.add(Convolution2D(20, 5, 5, border_mode="valid",input_shape=(60, 60, 3)))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    
    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="valid"))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    
    '''
    # set of FC => RELU layers
    model.add(Dense(500))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    '''
    # softmax classifier
    model.add(Dense(32))
    model.add(Dropout(drop))
    model.add(Activation("softmax"))
    return model
    
    
def main(img):   
        # Setting up the hyperparameters
    num_classes = 32
    #drop = 0.2
    
    
    # Initializing the model
    model = le_net(0.2)
    
    # Load the weights of the trained model
    model.load_weights('Lenet_all_marked.h5')
    #train_images = np.load('train_images_all_marked.npy')
    #train_labels = np.load('train_labels_all_marked.npy')
    #labels = train_labels
    
    
    
    
    # Specify the class names
    class_names = {0: 'AppleBlackRot', 1: 'AppleCedarAppleRust', 2: 'AppleHealthy', 3: 'AppleScab', 4: 'BlueberryHealthy',
                   5: 'CherryHealthy', 6: 'CherryPowderyMildew', 7: 'CornCommonRust', 8: 'CornNorthernLeafBlight',
                   9: 'GrapeBlackRot', 10: 'GrapeEsca', 11: 'GrapeHealthy', 12: 'GrapeLeafBlight',
                   13: 'PeachHealthy', 14: 'PepperBellBacterialSpot', 15: 'PepperBellHealthy',
                   16: 'PotatoEarlyBlight', 17: 'PotatoLateBlight', 18: 'RaspberryHealthy', 19: 'SoybeanHealthy',
                   20: 'SquashPowderyMildew', 21: 'StrawberryHealthy', 22: 'StrawberryLeafScorch', 23: 'TomatoBacterialSpot',
                   24: 'TomatoEarlyBlight', 25: 'TomatoHealthy', 26: 'TomatoLateBlight', 27: 'TomatoLeafMold',
                   28: 'TomatoMosaicVirus', 29: 'TomatoSeptoriaLeafSpot', 30: 'TomatoTargetSpot', 31: 'TomatoYellowLeafCurlVirus'}
    
    
    
    #Testing random image from Apple Scab that is testing for one image.
    #mean = np.mean(train_images,axis=(0, 1, 2, 3)) 
    #std = np.std(train_images,axis=(0, 1, 2, 3))   
    mean = 41.62853
    std = 52.244873
    start_time = time.time()
    #img = imresize(imread(os.getcwd()+"/AppleScab/apple_scab_(5).jpg", mode='RGB'),(60,60)).astype(np.float32)
    #img = imread(os.getcwd()+"/"+imgstr, mode='RGB')
    img = imresize(img,(60,60)).astype(np.float32)
    img = (img-mean)/(std+1e-7)
    img = np.expand_dims(img, axis=0)
    out = model.predict(img)
    
    finish_time = time.time()
    time_diff2 = (finish_time - start_time)
    output = "Class-" + str(np.argmax(out)) + " : " + class_names[np.argmax(out)]
    
    return output
    #print("time taken:"+str(time_diff2))
'''
if __name__ == '__main__':
    #path = 'D:\\major-project\\major\\Apple\\marked_code_files\\real_life_data\\apple_black_rot\\img6_marked'
    path = 'D:\\major-project\\major\\Apple\\marked_code_files\\test\\'
    for file in os.listdir(path):
        img = imread(path+file,mode='RGB')
        ans = main(img)
        print(file + "classified as: "+ans)
'''
