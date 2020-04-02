import numpy as np
import cv2 as cv
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
import imutils

def bad_foo(image, mask, path, i, j): #finding "nongland" regions
    mask[mask > 0] = 1
    num = 0
    
    while num < j: #doing actions until desired number of "nongland" regions were riched
        x = np.random.randint(0, image.shape[1] - 1) #generating random x,y,w,h
        y = np.random.randint(0, image.shape[0] - 1)        
        w = np.random.randint(1, 250)
        h = np.random.randint(1, 250)
        
        ny = max(1, min(image.shape[0] - 1, y + h)) #customizing to avoid index-is-out-of-range exception
        nx = max(1, min(image.shape[1] - 1, x + w))
        
        roi=np.where(mask[y:ny,x:nx] > 0, 0, 1) #mask for "nongland" area
        summ = np.sum(roi) #counting "nongland" area       
        if (np.float(summ) / (w * h) > 0.7) and (w * h > 400) and (np.float(w) / h > 0.33) and (np.float(w) / h < 3):
            cv.imwrite(path + "/" + str(i + num) + ".png", image[y:ny,x:nx]) #some cases to approve the region
            num = num + 1
        
    

def segmentation_foo(image, mask, path, i): #constructing "gland" samples
    mask[mask > 0] = 1
    label, classes = nd.label(mask) #separating "gland" regions
    
    for j in range(1, classes + 1):
        buf_mask = np.uint8(np.where(label == j, 1, 0)) #picking out concrete region
        contours, hierarchy = cv.findContours(buf_mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:] #finding it's countour in open-cv format
        idx = 0
        for cnt in contours:
            idx += 1
            x,y,w,h = cv.boundingRect(cnt) #making bounding rect
            for k in range(5):
                bx, by, bw, bh = np.random.randint(-15, 15, 4) #random modificaions
                
                y = max(0, min(image.shape[0] - 1, y + by)) #customizing to avoid index-is-out-of-range exception
                x = max(0, min(image.shape[1] - 1, x + bx))
                w = max(1, w + bw)
                h = max(1, h + bh)
                ny = max(1, min(image.shape[0] - 1, y + h))
                nx = max(1, min(image.shape[1] - 1, x + w))
                
                roi=np.where(label[y:ny,x:nx] > 0, 1, 0) #mask for "nongland" area
                summ = np.sum(roi) #counting "nongland" area  
                if (np.float(summ) / (w * h) > 0.7) and (w * h > 400) and (np.float(w) / h > 0.33) and (np.float(w) / h < 3):
                    cv.imwrite(path + "/" + str(i) + ".png", image[y:ny,x:nx]) #some cases to approve the region
                    i = i + 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
    	help="path to dataset of images")
    ap.add_argument("-m", "--masks", required=True,
    	help="path to dataset of masks")
    ap.add_argument("-g", "--glands", required=True,
    	help="path to save glands patterns")
    ap.add_argument("-ng", "--nonglands", required=True,
    	help="path to save non-glands patterns")
    
    args = vars(ap.parse_args())
    
    imagePaths = sorted(list(paths.list_images(args["images"]))) #reading images and annotations paths 
    maskPaths = sorted(list(paths.list_images(args["masks"])))
    
    i = np.int(0)
    for imagePath, maskPath in zip(imagePaths, maskPaths):
        print(imagePath + " - " + maskPath)
        img = cv.imread(imagePath)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        mask = cv.imread(maskPath)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        for k in range(4): # "flip" augmentations
            if k / 2 == 1:
                cv.flip(img, 0)
                cv.flip(mask, 0)
                
            if k % 2 == 1:
                cv.flip(img, 1)
                cv.flip(mask, 1)
            j = segmentation_foo(img, mask, args["glands"], i) #generating "gland" regions
            bad_foo(img, mask, args["nonglands"], i, j - i) #and "nongland" regions here
            i = copy.deepcopy(j)
    print(i)

if __name__ == "__main__":
    main()