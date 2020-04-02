import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import utils as np_utils
from keras import backend as keras
from skimage import exposure
from keras.callbacks import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from ML.model import mynet
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


#generates batches
def trainGenerator(batch_size,train_path, aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict) #making class object
    image_generator = image_datagen.flow_from_directory( #setting properties
        train_path,
        classes = None, #classes wil be set as array of subfolders names
        class_mode = "categorical", #to transorm them to 2D one-hot encoded labels
        color_mode = image_color_mode,
        target_size = target_size, #will be resized to this size
        batch_size = batch_size,
        save_to_dir = save_to_dir, #only augmented images will be saved
        save_prefix  = image_save_prefix,
        interpolation = "bilinear",
        seed = seed)
    return image_generator

def valGenerator(image_path, mask_path):
  imagePaths = sorted(list(paths.list_images(image_path)))
  maskPaths = sorted(list(paths.list_images(mask_path)))

  for(image, msk) in zip(imagePaths, maskPaths):
    img = io.imread(image, as_gray = True)
    mask = io.imread(msk, as_gray = True)

    img = np.reshape(img,img.shape+(1,))
    img = np.reshape(img,(1,)+img.shape)

    mask = np.reshape(mask,mask.shape+(1,))
    mask = np.reshape(mask,(1,)+mask.shape)

    img,mask = adjustData(img,mask)
    yield (img,mask)

def train(train_path, validation_path, epochs = 100, batch_size = 32, 
          checkpoint_file = './checkpoint.hdf5', statistic_folder = None, save_to_dir = None):
    
    epochs = int(epochs)
    batch_size = int(batch_size)
    
    data_gen_args = dict(vertical_flip=True,
                    horizontal_flip=True,
                    #brightness_range=(-0.05, 0.05),
                    rotation_range=10,
                    fill_mode='nearest')
    data_val_args = dict(rotation_range=10,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=True,
                    fill_mode='nearest')
    
    myGene = trainGenerator(batch_size, train_path, data_gen_args, save_to_dir = None)
    valGene = trainGenerator(8, validation_path, data_val_args, save_to_dir = None)
    
    model = mynet()
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss',
                                       verbose=1, save_best_only=True)
    
    early_stopping = EarlyStopping(monitor='loss', patience=7) #patience provides number of epocs befour this function will be activated
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001) #factor is a scaling fator to learning rate
    
    callback_list = [early_stopping, reduce_lr]
    if statistic_folder != None:
        csv_logger = CSVLogger(statistic_folder + '/history.csv')
        # tensorboard_logger = TensorBoard(log_dir= statistic_folder +'/tensorboard', histogram_freq=1, batch_size=32,
        #                                                                 write_graph=True, write_grads=False, write_images=False,
        #                                                                 embeddings_freq=0, embeddings_layer_names=None,
        #                                                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
        callback_list.append(csv_logger)
        
    model.fit_generator(myGene,steps_per_epoch=1000,epochs=epochs,
                        callbacks=callback_list, 
                        validation_data = valGene, validation_steps=400)

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=True,
	help="path to train dataset of images")
ap.add_argument("-v", "--validation", required=True,
	help="path to validation dataset of images")
ap.add_argument("-b", "--batchsize", required=False,
	help="number of images in one batch")
ap.add_argument("-e", "--epochs", required=False,
	help="number of epochs")
ap.add_argument("-c", "--checkpoint", required=False,
	help="path to checkpoint hdf5 file")
ap.add_argument("-stat", "--statistics", required=False,
	help="statistics folder")

args = vars(ap.parse_args())

train_args = dict(train_path = args["train"],
                  validation_path = args["validation"])

if args["epochs"] != None:
    train_args["epochs"] = args["epochs"]
if args["batchsize"] != None:
    train_args["batch_size"] = args["batchsize"]
if args["checkpoint"] != None:
    train_args["checkpoint_file"] = args["checkpoint"]
if args["statistics"] != None:
    train_args["statistic_folder"] = args["statistics"]
    

train(**train_args)

