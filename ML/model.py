import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as keras

LEARNING_RATE = 1e-3


def mynet(pretrained_weights = None,input_size = (256,256,1), learning_rate = LEARNING_RATE):
    start = Input(input_size)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(start)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    conv0 = BatchNormalization()(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    flatten = Flatten()(conv3)
    drop = Dropout(0.3)(flatten)
    dense = Dense(64, activation = 'relu', kernel_initializer = 'he_normal')(drop)
    dense = BatchNormalization()(dense)
    final = Dense(2, activation = 'sigmoid', kernel_initializer = 'he_normal')(dense)    

    model = Model(inputs = start, outputs = final)

    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def smallnet(pretrained_weights = None, input_size = (256,256,1), learning_rate = LEARNING_RATE):
    start = Input(input_size)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(start)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    conv0 = BatchNormalization()(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(8, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)
    
    flatten = Flatten()(pool2)
    dense = Dense(32, activation = 'relu', kernel_initializer = 'he_normal')(flatten)
    dense = BatchNormalization()(dense)
    final = Dense(2, activation = 'sigmoid', kernel_initializer = 'he_normal')(dense)    

    model = Model(inputs = start, outputs = final)

    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def densenet(pretrained_weights = None,input_size = (256,256,1), learning_rate = LEARNING_RATE):
    first = Input(input_size)
    conv0_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(first)
    conv0_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_1)
    batch0_1 = BatchNormalization()(conv0_1)

    conv0_2 = Conv2D(8, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch0_1)
    conv0_2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_2)
    cct0_12 = concatenate([conv0_1,conv0_2], axis = 3)
    batch0_2 = BatchNormalization()(cct0_12)

    conv0_3 = Conv2D(8, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch0_2)
    conv0_3 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_3)
    cct0_123 = concatenate([conv0_1,conv0_2, conv0_3], axis = 3)
    batch0_3 = BatchNormalization()(cct0_123)

    conv0_4 = Conv2D(8, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch0_3)
    conv0_4 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_4)
    cct0_1234 = concatenate([conv0_1,conv0_2, conv0_3, conv0_4], axis = 3)
    batch0_4 = BatchNormalization()(cct0_1234)

    conv0_5 = Conv2D(8, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch0_4)
    conv0_5 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_5)
    cct0_12345 = concatenate([conv0_1,conv0_2, conv0_3, conv0_4, conv0_5], axis = 3)
    batch0_5 = BatchNormalization()(cct0_12345)

    dec0 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch0_5)
    pool0 = MaxPooling2D(pool_size=(2, 2))(dec0)

    conv1_1 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    cct1_1 = concatenate([pool0,conv1_1], axis = 3)
    batch1_1 = BatchNormalization()(cct1_1)

    conv1_2 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch1_1)
    conv1_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    cct1_12 = concatenate([pool0,conv1_1,conv1_2], axis = 3)
    batch1_2 = BatchNormalization()(cct1_12)

    conv1_3 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch1_2)
    conv1_3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_3)
    cct1_123 = concatenate([pool0,conv1_1,conv1_2,conv1_3], axis = 3)
    batch1_3 = BatchNormalization()(cct1_123)

    conv1_4 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch1_3)
    conv1_4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_4)
    cct1_1234 = concatenate([pool0,conv1_1,conv1_2,conv1_3,conv1_4] , axis = 3)
    batch1_4 = BatchNormalization()(cct1_1234)

    dec1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch1_4)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dec1)

    conv2_1 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2_1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
    cct2_1 = concatenate([pool1,conv2_1], axis = 3)
    batch2_1 = BatchNormalization()(cct2_1)

    conv2_2 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch2_1)
    conv2_2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_2)
    cct2_12 = concatenate([pool1,conv2_1,conv2_2], axis = 3)
    batch2_2 = BatchNormalization()(cct2_12)

    conv2_3 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch2_2)
    conv2_3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_3)
    cct2_123 = concatenate([pool1,conv2_1,conv2_2,conv2_3], axis = 3)
    batch2_3 = BatchNormalization()(cct2_123)

    conv2_4 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch2_3)
    conv2_4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_4)
    cct2_1234 = concatenate([pool1,conv2_1,conv2_2,conv2_3,conv2_4] , axis = 3)
    batch2_4 = BatchNormalization()(cct2_1234)    

    dec2_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(batch2_4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dec2_1)
    
    dec2_2 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    dec2_3 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dec2_2)

    flatten = Flatten()(dec2_3)
    drop = Dropout(0.3)(flatten)
    dense = Dense(64, activation = 'relu', kernel_initializer = 'he_normal')(drop)
    dense = BatchNormalization()(dense)
    final = Dense(2, activation = 'sigmoid', kernel_initializer = 'he_normal')(dense)    

    model = Model(inputs = first, outputs = final)

    model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
