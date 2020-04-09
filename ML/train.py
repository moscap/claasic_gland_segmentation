import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from tensorflow.keras import backend as keras
from skimage import exposure
from tensorflow.keras import utils as np_utils
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import platform
import cv2
import os

from model import mynet, densenet
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

MIN_LR = 0.0000001
BASE_PATIENCE = 6
FACTOR = 0.2
MODEL_BASE_TYPE = 'std'
MODEL_DENSE_TYPE = 'dense'
BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
ROTATION_RANGE = 15
EPOCHS = 300

#generates batches
def myTrainGenerator(btch_size, data_x, data_y, aug_dict):
    image_datagen = ImageDataGenerator(**aug_dict) #making class object
    image_generator = image_datagen.flow(data_x, data_y, batch_size = btch_size)

    return image_generator

#starts traning
def Train(train_x, train_y, validation_x, validation_y, 
          model_type = MODEL_BASE_TYPE, epochs = EPOCHS, 
          batch_size = BATCH_SIZE, checkpoint_file = './checkpoint.hdf5', 
          history_file = 'history.csv', statistic_folder = None, save_to_dir = None):
    
    data_gen_args = dict(vertical_flip=True,
                         horizontal_flip=True,
                         rotation_range=ROTATION_RANGE,
                         fill_mode='nearest')
    data_val_args = dict(rotation_range= 5.0,
                         width_shift_range=0.0,
                         height_shift_range=0.0,
                         shear_range=0.0,
                         zoom_range=0.0,
                         horizontal_flip=True,
                         fill_mode='nearest')
    
    myGene = myTrainGenerator(batch_size, train_x, train_y, data_gen_args)
    valGene = myTrainGenerator(VAL_BATCH_SIZE, validation_x, validation_y, data_val_args)

    if(mdel_type == MODEL_BASE_TYPE):
        model = mynet()
    else:
        model = densenet()
        
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss',verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', patience=BASE_PATIENCE + 2) #ptiaence provides number of epocs befour this function will be activated
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=FACTOR, patience=BASE_PATIENCE, min_lr=MIN_LR) #factor is a scaling fator to learning rate
    csv_logger = CSVLogger(history_file)
    
    # tensorboard_logger = TensorBoard(log_dir='/content/drive/My Drive/Diploma/logs/tensorboard', histogram_freq=1, batch_size=batch_size,
    #                                                                 write_graph=True, write_grads=False, write_images=False,
    #                                                                 embeddings_freq=0, embeddings_layer_names=None,
    #                                                                 embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    model.fit_generator(myGene,steps_per_epoch=len(train_x) * 3 / BATCH_SIZE,
                        epochs=100, callbacks=[model_checkpoint, early_stopping, reduce_lr, csv_logger], 
                        validation_data = valGene, validation_steps=len(validation_x) * 3 / VAL_BATCH_SIZE)

#prepare data
def train(train_path, validation_path, model_type = MODEL_BASE_TYPE, epochs = EPOCHS, 
          batch_size = BATCH_SIZE, checkpoint_file = './checkpoint.hdf5', 
          history_file = 'history.csv', statistic_folder = None, save_to_dir = None):
    
    epochs = int(epochs)
    batch_size = int(batch_size)

    with np.load('../data.npz') as data:
        tr_x = data['tr_x']
        tr_y = data['tr_y']
        val_x = data['val_x']
        val_y = data['val_y']
        
        print(np.asarray(tr_y).shape)    
        
        Train(tr_x, tr_y, val_x, val_y, model_type, epochs, batch_size, checkpoint_file , 
              history_file, statistic_folder, save_to_dir)

#argparse
def main():
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
    ap.add_argument("-hist", "--history", required=False,
    	help="name of cvsv file with trainig history")
    ap.add_argument("-stat", "--statistics", required=False,
    	help="statistics folder")
    ap.add_argument("-type", "--type", required=False,
    	help="model type: std or dense")
    
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
    if args["type"] != None:
        train_args["model_type"] = args["type"]
    if args["history"] != None:
        train_args["history_file"] = args["history"]

    train(**train_args)

if __name__ == "__main__":
    main()