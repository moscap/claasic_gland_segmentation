import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage as nd 
import skimage.measure as skm
import skimage.morphology as skmorf
import skimage.transform as skt
import skimage.draw as skd
from skimage.util import invert
import skimage.feature as skf
import copy
import os
import math

import Classic.graph as graph

PIXEL_MAX_VALUE = 255
ANGLE_PIECE = 0.125
DIST_TRESHOLD = 30
DIST_TRANSFORM_TRESHOLD = 0.3
MASK_LOWER_TRESHOLD = 0.15
MASK_UPPER_TRESHOLD = 0.5
MASK_MEDIAN_FILTER_SIZE = 3
GAUSS_GRADIENT_SIGMA = 2
PERIMETER_EROSION_FILTER_SHAPE = (3,3)
PIXEL_REVIEW_MIN = -6
PIXEL_REVIEW_MAX = 7
AREA_TRESHOLD = 0.1
BOUNDING_GLAND_CORRECTION = 0.07
AXIS_Y = 0
AXIS_X = 1
CLEAR_ITERATIONS = 10

#finding gaussian gradient filter of image
def find_gaussian_gradient(image):
    gaussy = nd.gaussian_filter1d(image, GAUSS_GRADIENT_SIGMA, axis = AXIS_Y, order = 1 )
    gaussy = nd.gaussian_filter1d(gaussy, GAUSS_GRADIENT_SIGMA, axis = AXIS_X, order = 0 )
    
    gaussx = nd.gaussian_filter1d(image, GAUSS_GRADIENT_SIGMA, axis = AXIS_X, order = 1 )
    gaussx = nd.gaussian_filter1d(gaussx, GAUSS_GRADIENT_SIGMA, axis = AXIS_Y, order = 0 )
    
    gauss = np.arctan2(gaussx, gaussy) / np.pi
    return gauss

#generationg dilation kernel with missing corners
def init_dilate_kernel(size_x, size_y):
    dilate_kernel = np.ones((size_x,size_y), dtype = np.uint8)
    
    dilate_kernel[0, -1] = 0
    dilate_kernel[-1, -1] = 0
    dilate_kernel[0, 0] = 0
    dilate_kernel[-1, 0] = 0
    
    return dilate_kernel  
    
#returns perimeter of    
def get_perimeter(image):
    erosed = nd.binary_erosion(image, np.ones(PERIMETER_EROSION_FILTER_SHAPE))
    erosed[0, :] = 1.0
    erosed[-1, :] = 1.0
    erosed[:, 0] = 1.0
    erosed[:, -1] = 1.0
    perim = image - erosed
    return perim

#deletes unnecessary pixels from mask
def clear(mask, angles, perim, perim_mask):
    #получаем рассматриваемы точки периметра
    bounds = np.argwhere((perim[1:-1, 1:-1]  > 0) * (perim_mask[1:-1, 1:-1] == 0)) 
    bounds = bounds + 1
        
    for bound in bounds:
        angle = angles[bound[AXIS_Y], bound[AXIS_X]]
        pixel = mask[bound[AXIS_Y], bound[AXIS_X]]
        
        #смотрим на угол, проверяем необходимость удаления из маски     
        if angle <= -7 * ANGLE_PIECE or angle >= 7 * ANGLE_PIECE or (angle >= -ANGLE_PIECE and angle <= ANGLE_PIECE):
            if max(mask[bound[AXIS_Y], bound[AXIS_X] + 1], mask[bound[AXIS_Y], bound[AXIS_X] - 1]) > pixel:
                mask[bound[AXIS_Y], bound[AXIS_X]] = 0
        elif (angle > -7 * ANGLE_PIECE and angle <= -5 * ANGLE_PIECE) or (angle > ANGLE_PIECE and angle <= 3 * ANGLE_PIECE):
            if max(mask[bound[AXIS_Y] - 1, bound[AXIS_X] - 1], mask[bound[AXIS_Y] + 1, bound[AXIS_X] + 1]) > pixel:
                mask[bound[AXIS_Y], bound[AXIS_X]] = 0
        if (angle < 7 * ANGLE_PIECE and angle > 5 * ANGLE_PIECE) or (angle > -3 * ANGLE_PIECE and angle < -ANGLE_PIECE):
            if max(mask[bound[AXIS_Y] + 1, bound[AXIS_X] - 1], mask[bound[AXIS_Y] - 1, bound[AXIS_X] + 1]) > pixel:
                mask[bound[AXIS_Y], bound[AXIS_X]] = 0
        else:
            if max(mask[bound[AXIS_Y] + 1, bound[AXIS_X]], mask[bound[AXIS_Y] - 1, bound[AXIS_X]]) > pixel:
                mask[bound[AXIS_Y], bound[AXIS_X]] = 0
    
    return mask

#reviews certain pixel for the need of isolation
def pixel_review(pixel, mask, perim_mask, prev_label):
    maxx = -np.Inf
    minn = np.Inf
    for m in range(PIXEL_REVIEW_MIN,PIXEL_REVIEW_MAX): #просматриваем "окном" соотвествующего размера
        for k in range(PIXEL_REVIEW_MIN,PIXEL_REVIEW_MAX):
            buf = prev_label[min(prev_label.shape[AXIS_Y] - 1, max(0, pixel[AXIS_Y] + m)), 
                             min(prev_label.shape[AXIS_X] - 1, max(0, pixel[AXIS_X] + k))]
            if(buf > 0): 
                maxx = max(maxx, buf)
                minn = min(minn, buf)
        #если встречаются две разные области в одном окне => проводим разделитель               
        if (maxx != minn) and (minn != np.Inf):
            perim_mask[pixel[AXIS_Y], pixel[AXIS_X]] = 1.0
            mask[pixel[AXIS_Y], pixel[AXIS_X]] = 1.0
            
    return mask, perim_mask

#one step in clearing process
def clearing_step(mask, perim_mask, prev_label, total_classes):
    bi_mask = np.where(mask[:,:] > 0, 1.0, 0.0)#получили границы
    angles = find_gaussian_gradient(bi_mask);#получили градиент на границах
        
    erosed = nd.binary_erosion(bi_mask, np.ones(PERIMETER_EROSION_FILTER_SHAPE))#провели эрозию
    erosed[0, :] = 1.0
    erosed[-1, :] = 1.0
    erosed[:, 0] = 1.0
    erosed[:, -1] = 1.0#исключили границы картинки
    perim = get_perimeter(bi_mask)#поулучили периметр 
    
    label, classes = nd.label(1.0 - erosed)#сегментация расширенной картинки
    if(classes != total_classes):
        full_perim = np.argwhere((perim > 0)) #смотрим весь периметр 
        for pixel in full_perim:
            if(perim_mask[pixel[AXIS_Y], pixel[AXIS_X]] > 0):#если этот пиксель помечен в карте периметра не рассматриваем
                continue
            
            pixel_review(pixel, mask, perim_mask, prev_label)
        
    #cv.imwrite("../masks/mmask" + str(j) + ".png", mask * 255 )
        
    label[perim_mask > 0] = 0 #немного магии нужной для корректного обновления сегментации
    label[label > 0] = 1
    label, classes = nd.label(label)
    prev_label = label        
        
    mask = clear(mask, angles, perim, perim_mask) #удаляем "ненужные" пиксели из маски
                     
    opening = copy.deepcopy(mask) #тоже какая то магия
    opening[opening > 0] = 1
    opening = skmorf.binary_opening(opening, init_dilate_kernel(3,3))
    opening[perim_mask > 0] = 1
    mask = mask * opening
                         
    return mask, perim_mask, prev_label
        
def clear_mask(mask):
    st_bi_mask = np.where(mask[:,:] > 0, 0.0, 1.0)#начальная бинаризация    
    prev_label, total_classes = nd.label(st_bi_mask)#начальная сегментация
    perim_mask = np.zeros(mask.shape)
    
    for j in range(CLEAR_ITERATIONS):
        print(j)
        mask, perim_mask, prev_label = clearing_step(mask, perim_mask, prev_label, total_classes)                     
                    
    cl_mask = nd.median_filter(nd.gaussian_filter(mask, 2), 5) * 10000 
    cl_mask = np.where(cl_mask < 128, 0,1)
    mask[mask > 0] = 1
    mask = skmorf.binary_opening(mask, init_dilate_kernel(5,5))
    mask[perim_mask > 0] = 1
    
    return mask
    

#making some basic improvements to mask
def mask_post_process(mask):    
    mask = np.double(mask)
    mask = mask / np.max(np.max(mask))
    
    dist_mask = np.where(mask < DIST_TRANSFORM_TRESHOLD, 1, 0)
    distances = nd.distance_transform_edt(dist_mask)
        
    mask[distances > DIST_TRESHOLD] = 0
    
    mask = nd.median_filter(mask, MASK_MEDIAN_FILTER_SIZE)
    mask[mask < MASK_LOWER_TRESHOLD] = 0
    mask[mask > MASK_UPPER_TRESHOLD] = 1   
    
    cleared = clear_mask(copy.deepcopy(mask))    
        
    return cleared

#posttprocessing watershed
def watershed_post_process(mask):
    watershed = np.where(mask == 0, 1, 0)
    watershed = nd.label(watershed)[0]
    watershed[mask > 0] = -1
    watershed += 1
    #watershed = nd.watershed_ift(np.uint16(np.where(mask > 0, 1, 0)), watershed)
    watershed = np.uint16(skmorf.watershed(mask, watershed, watershed_line = True))
    watershed = np.where(watershed > 1, PIXEL_MAX_VALUE, 0)
    
    label, classes = nd.label(watershed)    
    maxx = 0
    
    for i in range(1, classes):
        summ = sum(sum(np.where(watershed == i, 1, 0)))
        if summ > maxx:
            maxx = summ;
            
    for i in range(1, classes):
        segment = np.where(watershed == i, 1, 0)
        summ = sum(segment[:, -1]) + sum(segment[:, 0]) + sum(segment[-1, :]) + sum(segment[0, :])
        is_bordered = (summ) > 0 
        summ = sum(sum(segment))
        if summ <= (AREA_TRESHOLD - (BOUNDING_GLAND_CORRECTION if is_bordered else 0)) * maxx:
            watershed[watershed == i] = 0
            
    fig, axes = plt.subplots(ncols=2, figsize=(20, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(watershed * PIXEL_MAX_VALUE, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Изображение')
    plt.tight_layout()
    plt.show()  
    
    return watershed, classes

if __name__ == "__main__":
    print("This module can't be executed independently")