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
GENERATING_MAX_SIZE = 200
GENERATING_MIN_SIZE = 1
BIAS_PORTION = 0.1
BIG_SHIFT_STEP = 3
ROTATION_STEP = 3
STEPS_TO_TRY = 10
AUG_STEPS = 3
GLAND = 'glands'
NONGLAND = 'nonglands'
MISS = 'miss'
TRAIN = 'train'

AXIS_X = 1
AXIS_Y = 0
            
def check_sample_with_bias(image, label, mask, path, x, y, w, h, bx, by, bw, bh, i, tipe):  
    img = image
    
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
            np.float(w) / h > 1.0 / RATIO_TRESHOLD and
            np.float(w) / h < RATIO_TRESHOLD
        ):
        patch = cv.resize(img[y:ny,x:nx], RESIZE_SHAPE, interpolation = cv.INTER_LINEAR)
        mask_patch = cv.resize(mask[y:ny,x:nx], RESIZE_SHAPE, interpolation = cv.INTER_LINEAR)
        cv.imwrite(path + "/" + tipe + "/" + str(i) + ".png", patch) #some cases to approve the region
        cv.imwrite(path + "/mask/" + tipe + "/" + str(i) + ".png", mask_patch * 255) #some cases to approve the region
        i = i + 1
    return i


def test_region(patch, gt):
    mask = copy.deepcopy(gt)
    patch = copy.deepcopy(patch)
    mask[patch == 0] = 0
    patch[patch > 0] = 1
    
    patch_sum = np.sum(patch)
    mask_sum = np.sum(mask)
    
    if np.float(mask_sum) / patch_sum > 0.85 and mask[mask.shape[AXIS_Y] // 2, mask.shape[AXIS_X] // 2] > 0:
        return GLAND
    elif np.float(mask_sum) / patch_sum < 0.15 and mask[mask.shape[AXIS_Y] // 2, mask.shape[AXIS_X] // 2] == 0:
        return NONGLAND
    else:
        return MISS
        
    

def segmentation_foo(image, mask, patch, path, i, mode): #constructing "gland" samples    
    mask[mask > 0] = 1 #mask pereparation
    patch[patch > 0] = 1 #patch preparation
    label, classes = nd.label(patch) #separating regions
    
    for j in range(1, classes + 1):
        patch_mask = np.uint8(np.where(label == j, 1, 0)) #picking out concrete region        
        contours, hierarchy = cv.findContours(patch_mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:] #finding it's countour in open-cv format
        idx = 0
        for cnt in contours:
            idx += 1
            x,y,w,h = cv.boundingRect(cnt) #making bounding rect
            tipe = test_region(patch_mask[y:y+h, x:x+w], mask[y:y+h, x:x+w]) #now we knew region type
            
            if tipe == MISS: #next step
                continue
            
            if mode ==  TRAIN:
                bias = np.floor(BIAS_PORTION * max(w, h)) #augmentation lower
                if bias == 0: #too small region
                    continue
                
                bx, by, bw, bh = np.random.randint(-bias, bias, 4) #random modificaions                
                i = check_sample_with_bias(image, label, mask, path, x, y, w, h, bx, by, bw, bh, i, tipe)                
                
                if i % BIG_SHIFT_STEP == 0  and tipe == GLAND:
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
                            
                        k = check_sample_with_bias(image, label, mask, path, x, y, w, h, bx, by, bw, bh, i, tipe)
                        steps = steps + 1 
                    i = copy.deepcopy(k)  
            else:
                i = check_sample_with_bias(image, label, mask, path, x, y, w, h, 0, 0, 0, 0, i, tipe)
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
    	help="path to dataset of images")
    ap.add_argument("-m", "--masks", required=True,
    	help="path to dataset of masks")
    ap.add_argument("-p", "--patches", required=True,
    	help="path to dataset of masks")
    ap.add_argument("-mode", "--mode", required=True,
    	help="mode: train val or test")
    ap.add_argument("-s", "--save", required=True,
    	help="path to saving folder.\n folder should contain 'glands' and 'nonglands subfolders")
    
    args = vars(ap.parse_args())
    
    imagePaths = sorted(list(paths.list_images(args["images"]))) #reading images and annotations paths 
    maskPaths = sorted(list(paths.list_images(args["masks"])))
    patchesPaths = sorted(list(paths.list_images(args["patches"])))
    
    i = 0
    for imagePath, maskPath, patchPath in zip(imagePaths, maskPaths, patchesPaths):
        print(imagePath + " - " + maskPath)
        img = cv.imread(imagePath)
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        patch = cv.imread(patchPath)
        patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        
        mask = cv.imread(maskPath)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        
        if args["mode"] == TRAIN:
            for k in range(AUG_STEPS): # "flip" augmentations     
                angle = np.random.randint(0, len(ROTATION_ANGLES))
                angle = ROTATION_ANGLES[angle]
                
                img = imutils.rotate_bound(img, angle)
                mask = imutils.rotate_bound(mask, angle)
                patch = imutils.rotate_bound(patch, angle)
                    
                i = segmentation_foo(img, mask, patch, args["save"], i, args["mode"]) #generating regions
        else:
            i = segmentation_foo(img, mask, patch, args["save"], i, args["mode"])
            
    print(i)

if __name__ == "__main__":
    main()