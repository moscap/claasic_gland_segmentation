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

ROTATION_ANGLES = [0, 90, 180, 270]
MIN_REGION_AREA_RATIO = 0.7
REGION_AREA_TRESHOLD = 400
RESIZE_SHAPE = (256, 256)
RATIO_TRESHOLD = 3.0
GENERATING_MAX_SIZE = 250
GENERATING_MIN_SIZE = 1
BIAS_PORTION = 0.1
BIG_SHIFT_STEP = 3
ROTATION_STEP = 3
STEPS_TO_TRY = 10
AUG_STEPS = 10

AXIS_X = 1
AXIS_Y = 0

def bad_foo(image, mask, path, i, j): #finding "nongland" regions
    mask[mask > 0] = 1
    num = 0
    
    img = copy.deepcopy(image)
    img[mask > 0] = 0
    
    while num < j: #doing actions until desired number of "nongland" regions were riched
        if num % ROTATION_STEP == 0:
            angle = np.random.randint(0, len(ROTATION_ANGLES))
            angle = ROTATION_ANGLES[angle]
            img = imutils.rotate_bound(img, angle)
            mask = imutils.rotate_bound(mask, angle)
    
        x = np.random.randint(0, image.shape[AXIS_X] - 1) #generating random x,y,w,h
        y = np.random.randint(0, image.shape[AXIS_Y] - 1)        
        w = np.random.randint(GENERATING_MIN_SIZE, GENERATING_MAX_SIZE)
        h = np.random.randint(GENERATING_MIN_SIZE, GENERATING_MAX_SIZE)
        
        ny = max(1, min(image.shape[AXIS_Y] - 1, y + h)) #customizing to avoid index-is-out-of-range exception
        nx = max(1, min(image.shape[AXIS_X] - 1, x + w))
        
        roi=np.where(mask[y:ny,x:nx] > 0, 0, 1) #mask for "nongland" area
        summ = np.sum(roi) #counting "nongland" area       
        if (
                np.float(summ) / (w * h) > MIN_REGION_AREA_RATIO and 
                w * h > REGION_AREA_TRESHOLD and 
                np.float(w) / h > 1 / RATIO_TRESHOLD and
                np.float(w) / h < RATIO_TRESHOLD
            ):
            patch = cv.resize(img[y:ny,x:nx], RESIZE_SHAPE, interpolation = cv.INTER_LINEAR)
            cv.imwrite(path + "/" + str(i + num) + ".png", patch) #some cases to approve the region
            num = num + 1
            
def check_sample_with_bias(image, label, path, x, y, w, h, bx, by, bw, bh, i):  
    img = copy.deepcopy(image)
    img[label == 0] = 0
    
    y = max(0, min(image.shape[AXIS_Y] - 1, y + by)) #customizing to avoid index-is-out-of-range exception
    x = max(0, min(image.shape[AXIS_X] - 1, x + bx))
    w = max(1, w + bw)
    h = max(1, h + bh)
    ny = max(1, min(image.shape[AXIS_Y] - 1, y + h))
    nx = max(1, min(image.shape[AXIS_X] - 1, x + w))
                
    roi=np.where(label[y:ny,x:nx] > 0, 1, 0) #mask for "nongland" area
    summ = np.sum(roi) #counting "nongland" area  
    if (
            np.float(summ) / (w * h) > MIN_REGION_AREA_RATIO and 
            w * h > REGION_AREA_TRESHOLD and 
            np.float(w) / h > 1 / RATIO_TRESHOLD and
            np.float(w) / h < RATIO_TRESHOLD
        ):
        patch = cv.resize(img[y:ny,x:nx], RESIZE_SHAPE, interpolation = cv.INTER_LINEAR)
        cv.imwrite(path + "/" + str(i) + ".png", patch) #some cases to approve the region
        i = i + 1
    return i
        
    

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
            
            bias = np.floor(BIAS_PORTION * max(w, h))
            if bias == 0:
                continue
            
            bx, by, bw, bh = np.random.randint(-bias, bias, 4) #random modificaions                
            i = check_sample_with_bias(image, label, path, x, y, w, h, bx, by, bw, bh, i)                
            
            if i % BIG_SHIFT_STEP == 0:
                k = copy.deepcopy(i)      
                steps = 0
                while k == i and steps < STEPS_TO_TRY:
                    small_bias = max(bias // 3, 1)
                    bx, by = np.random.randint(-small_bias, small_bias, 2) #random modificaions for big shift
                    
                    if w > h:
                        bw = np.random.randint(-7 * bias, -5 * bias)
                        bh = np.random.randint(-small_bias, 0)
                    else:
                        bh = np.random.randint(-7 * bias, -5 * bias)
                        bw = np.random.randint(-small_bias, 0)
                        
                    k = check_sample_with_bias(image, label, path, x, y, w, h, bx, by, bw, bh, i)
                    steps = steps + 1 
                    
                i = copy.deepcopy(k)            
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
    
    i = 0
    for imagePath, maskPath in zip(imagePaths, maskPaths):
        print(imagePath + " - " + maskPath)
        img = cv.imread(imagePath)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        mask = cv.imread(maskPath)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        for k in range(AUG_STEPS): # "flip" augmentations     
            angle = np.random.randint(0, len(ROTATION_ANGLES))
            angle = ROTATION_ANGLES[angle]
            img = imutils.rotate_bound(img, angle)
            mask = imutils.rotate_bound(mask, angle)
                
            j = segmentation_foo(img, mask, args["glands"], i) #generating "gland" regions
            bad_foo(img, mask, args["nonglands"], i, j - i) #and "nongland" regions here
            i = copy.deepcopy(j)
            
    print(i)

if __name__ == "__main__":
    main()