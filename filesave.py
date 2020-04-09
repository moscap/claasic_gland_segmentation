import skimage.io as io
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import utils as np_utils
from imutils import paths
import numpy as np
import cv2
import os

SHAPE_CORRECTION = (1,)
GLANDS = 'glands'
NONGLANDS = 'nonglands'
NUM_CLASSES = 2

def Load(samples_x, samples_y, path, num_classes = 2, as_gray = True):
    imagePaths = sorted(list(paths.list_images(path + '/glands')))
    additional = sorted(list(paths.list_images(path + '/nonglands'))) 
    imagePaths.extend(additional)                    

    for imgPath in imagePaths:
        img = io.imread(imgPath,as_gray = as_gray)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)
        img = np.reshape(img, img.shape + SHAPE_CORRECTION)

        cl = os.path.dirname(imgPath).split('/')[-1]
        if cl == GLANDS:
             cl = 0
        else:
             cl = 1

        samples_x.append(img)
        samples_y.append(cl)
    return samples_x, samples_y

tr_x = []
tr_y = []
val_x = []
val_y = []
    
val_x, val_y = Load(val_x, val_y, '../NN/val')
tr_x, tr_y = Load(tr_x, tr_y, '../NN/train')    
print("Detected " + str(len(tr_x) + len(tr_y)) + " images in train and " + str(len(val_x) + len(val_y))  + "images in validation.")
    
tr_y = np_utils.to_categorical(tr_y, NUM_CLASSES)
val_y = np_utils.to_categorical(val_y, NUM_CLASSES)

print(np.asarray(tr_x).shape, tr_y.shape)

np.savez('data', tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y)  