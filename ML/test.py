import skimage.io as io
from imutils import paths
import numpy as np
import argparse
import cv2 as cv
import os
from ML.model import mynet, densenet, smallnet
import platform

NONGLANDS = 'nonglands'
GLANDS = 'glands'
RESIZE_SHAPE = (256, 256)
PERSENTS = 100
SHAPE_CORRECTION = (1,)
VERBOSE = 1

LINUX = 'Linux'
WINDOWS = 'Windows'

MODEL_BASE_TYPE = 'std'
MODEL_DENSE_TYPE = 'dense'
MODEL_SMALL_TYPE = 'small'

def fileTestGenerator(test_path, as_gray = False):
    imagePaths = sorted(list(paths.list_images(test_path)))
    for imgPath in imagePaths:
        img = io.imread(imgPath, as_gray = as_gray)
        img = cv.resize(img, RESIZE_SHAPE, interpolation = cv.INTER_LINEAR)
        #img = np.reshape(img, img.shape + SHAPE_CORRECTION)
        img = np.reshape(img, SHAPE_CORRECTION + img.shape)
        yield img
        
def containerTestGenerator(samples):
    for sample in samples:
        img = cv.resize(sample, RESIZE_SHAPE, interpolation = cv.INTER_LINEAR)
        # img = np.reshape(img, img.shape + SHAPE_CORRECTION)
        img = np.reshape(img, SHAPE_CORRECTION + img.shape)
        yield img
        
def predict(testGene, weights_path, lenth, mode):
    if(mode == MODEL_BASE_TYPE):
        model = mynet(input_size = (256, 256, 3))
    elif(mode == MODEL_DENSE_TYPE):
        model = densenet(input_size = (256, 256, 3))
    else:
        model = smallnet(input_size = (256, 256, 3), learning_rate = 0.01)
        
    model.load_weights(weights_path)
    results = model.predict_generator(testGene, lenth, verbose=VERBOSE)
    return results
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
    	help="path to input dataset of images")
    ap.add_argument("-w", "--weights", required=True,
    	help="path to hdf5 weigts of model")
    ap.add_argument("-type", "--type", required=False,
    	help="model type: std, dense or small")
    args = vars(ap.parse_args())
            
    testGene = fileTestGenerator(args["dataset"])
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    results = predict(testGene, args["weights"], len(imagePaths), args["type"])
    
    right = 0
    
    for result, path in zip(results, imagePaths):
      ans = 1 if result > 0.5 else 0 #np.argmax(result)
      pred = os.path.dirname(path).split('\\' if platform.system() == WINDOWS else '/')[-1]
      
      if pred == NONGLANDS and ans == 0:
          right = right + 1
      if pred == GLANDS and ans == 1:
          right = right + 1
      
    print(right * PERSENTS / len(imagePaths))

if __name__ == "__main__":
    main()
