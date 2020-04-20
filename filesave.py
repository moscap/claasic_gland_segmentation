import skimage.io as io
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import utils as np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

SHAPE_CORRECTION = (1,)
GLANDS = 'glands'
NONGLANDS = 'nonglands'
NUM_CLASSES = 2

def Load(samples_x, samples_y, path, num_classes = 2, as_gray = False):
    imagePaths = sorted(list(paths.list_images(path + '/glands')))
    additional = sorted(list(paths.list_images(path + '/nonglands'))) 
    imagePaths.extend(additional)                    

    for imgPath in imagePaths:
        img = io.imread(imgPath,as_gray = as_gray)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)
        # img = np.reshape(img, img.shape + SHAPE_CORRECTION)

        cl = os.path.dirname(imgPath).split('/')[-1]
        if cl == GLANDS:
             cl = 1.0
        else:
             cl = 0.0

        samples_x.append(img)
        samples_y.append(cl)
    return samples_x, samples_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--validation", required=True,
    	help="path to validation folder")
    ap.add_argument("-t", "--train", required=True,
    	help="path to train folder")
    ap.add_argument("-s", "--save", required=True,
    	help="path to saving folder")
    ap.add_argument("-n", "--name", required=True,
    	help="npz archiv name without extention")
    
    args = vars(ap.parse_args())
    
    tr_x = []
    tr_y = []
    val_x = []
    val_y = []
        
    val_x, val_y = Load(val_x, val_y, args["validation"])
    tr_x, tr_y = Load(tr_x, tr_y, args["train"])    
    print("Detected " + str(len(tr_x)) + " images in train and " + str(len(val_x))  + " images in validation.")
        
    # tr_y = np_utils.to_categorical(tr_y, NUM_CLASSES)
    # val_y = np_utils.to_categorical(val_y, NUM_CLASSES)
    
    print(np.asarray(tr_x).shape, np.asarray(tr_y).shape, "- train")
    print(np.asarray(val_x).shape, np.asarray(val_y).shape, "- validation")
    
    np.savez(args["save"] + '/' + args["name"] + '.npz', tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y)  

if __name__ == "__main__":
    main()
