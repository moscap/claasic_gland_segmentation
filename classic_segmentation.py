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
import skimage.io as io
import random
import os

import ML.test as ml
import Classic.graph as graph
import Classic.postprocess as pp
    
def ML_foo(watershed, image):
    mask = np.where(watershed > 0, 1, 0)
    label, classes = nd.label(mask)
    
    samples = []    
    
    for j in range(1, classes + 1):
        buf_mask = np.uint8(np.where(label == j, 1, 0)) #picking out concrete region
        contours, hierarchy = cv.findContours(buf_mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:] #finding it's countour in open-cv format
        idx = 0
        for cnt in contours:
            idx += 1
            x,y,w,h = cv.boundingRect(cnt) #making bounding rect
            samples.append(image[y:y+h, x:x+w])
            
            
    predicted = ml.predict(ml.containerTestGenerator(samples), './ML/32B3L.hdf5', len(samples))
    for idx, pred in enumerate(predicted):
        if np.argmax(pred) == 1:
            label[label == idx + 1] = 0
    return label

#post-processing bacis prediction results
def post_process(mask, image, filename):
    print("Post processing is already running..." )
    
    new_mask = pp.mask_post_process(copy.deepcopy(mask))  
    mmask = copy.deepcopy(mask * new_mask)         
    watershed, classes = pp.watershed_post_process(mmask)
    watershed = ML_foo(watershed, image)
      
    fig, axes = plt.subplots(ncols=2, figsize=(20, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(mask, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Маска до очистки')
    ax[1].imshow(mask * new_mask, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Очищенная маска')  
    plt.tight_layout()
    plt.savefig("../masks/" + filename + ".png")
    plt.show()
    
    c_watershed = cv.cvtColor(np.uint8(watershed), cv.COLOR_GRAY2BGR) 
    
    for i in range(1, classes + 1):
        c_watershed[watershed == i] = [i, 255 - i, 255]
    
    
    cv.imwrite("../masks/mmask.png" , new_mask * 255)
    
    return watershed, new_mask
        

#basic function colltiolling prediction process
def make_sp(image, rays, dots, radius, filename):      
    mask = graph.find_bound_mask(image, rays, dots, radius)    
    
    watershed, new_mask = post_process(mask, image, filename)
    watershed[watershed > 0] = 1
#    kernel = init_dilate_kernel(5,5)
#    watershed = nd.binary_dilation(watershed, kernel)   
    
    
    new_im = cv.cvtColor(np.uint8(image), cv.COLOR_GRAY2BGR)    
    new_im[watershed > 0, 1:2] = 0
     
    
    #cv.imwrite("./wtshd.png", watershed)
          
    return new_im
        
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
args = vars(ap.parse_args())

images = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
print(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    data = cv.imread(imagePath)
    data = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    data = np.float64(data)
    # data = cv.resize(data, (726, 544), interpolation = cv.INTER_AREA)
    images.append(data)

if not images[0] is None:
   i = 1
   for img in images:
       start_time = time.time()       
       print(img.shape)
       
       img = make_sp(img, 10, 8, 120, str(i))       
       cv.imwrite("../res/" + str(i) + ".png" , img * 255)
       
       i = i + 1       
       print(time.time() - start_time)