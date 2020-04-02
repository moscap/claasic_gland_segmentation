
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras




def mynet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
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
    conv3 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    flatten = Flatten()(conv3)
    drop = Dropout(0.3)(flatten)
    dense = Dense(64, activation = 'relu', kernel_initializer = 'he_normal')(drop)
    dense = BatchNormalization()(dense)
    final = Dense(2, activation = 'sigmoid', kernel_initializer = 'he_normal')(dense)    

    model = Model(input = inputs, output = final)

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
