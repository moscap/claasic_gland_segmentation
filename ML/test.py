import skimage.io as io
from imutils import paths
import numpy as np
import argparse
import cv2 as cv
import os
from model import mynet
import platform

NONGLANDS = 'nonglands'
GLANDS = 'glands'

LINUX = 'Linux'
WINDOWS = 'Windows'

def fileTestGenerator(test_path, as_gray = True):
    imagePaths = sorted(list(paths.list_images(test_path)))
    for imgPath in imagePaths:
        img = io.imread(imgPath,as_gray = as_gray)
        img = cv.resize(img, (256, 256), interpolation = cv.INTER_LINEAR)
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
def containerTestGenerator(samples):
    for sample in samples:
        img = cv.resize(sample, (256, 256), interpolation = cv.INTER_LINEAR)
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
def predict(testGene, weights_path, lenth):
    model = mynet()
    model.load_weights(weights_path)
    results = model.predict_generator(testGene, lenth, verbose=1)
    return results
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
    	help="path to input dataset of images")
    ap.add_argument("-w", "--weights", required=True,
    	help="path to hdf5 weigts of model")
    args = vars(ap.parse_args())
            
    testGene = fileTestGenerator(args["dataset"])
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    results = predict(testGene, args["weights"], len(imagePaths))
    
    right = 0
    
    for result, path in zip(results, imagePaths):
      ans = np.argmax(result)
      pred = os.path.dirname(path).split('\\' if platform.system() == WINDOWS else '/')[-1]
      
      if pred == NONGLANDS and ans == 1:
          right = right + 1
      if pred == GLANDS and ans == 0:
          right = right + 1
      
    print(right * 100 / len(imagePaths))

if __name__ == "__main__":
    main()
