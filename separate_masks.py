import numpy as np
import cv2 as cv
import networkx as nx
import argparse
import matplotlib.pyplot as plt
from imutils import paths
from scipy import ndimage as nd 
import time
import math
import skimage.measure as skm
import skimage.morphology as skmorf
import skimage.transform as skt
import skimage.draw as skd
from skimage.util import invert
import skimage.feature as skf
import copy
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-s", "--save", required=True,
	help="path to save dataset of images")
args = vars(ap.parse_args())

images = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
i = 0
for imagePath in imagePaths: #simply 
    data = cv.imread(imagePath)
    if i % 2  == 0: #if image saving as image
        cv.imwrite(args["save"] + "/img" + os.path.splitext(imagePath)[0] + ".png", data)
    else: #if mask remmoving anno injection and then saving as mask
        cv.imwrite(args["save"] + "/mask" + os.path.basename(imagePath)[:-9] + ".png", data)
    i = i + 1
    
    
       
