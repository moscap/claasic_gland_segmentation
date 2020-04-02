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

Calc_g = nx.DiGraph()

#parsing the way and adding changes to mask
def parse_way(mask_region, way, frag_map, rays):
    for i in range(rays):
        mask_region[frag_map[way[i]]] += 1
    return mask_region
        
#function that findes way with mimimal sum of node weights 
def find_min_way(rays, dots):
    global Calc_g
    ans = nx.dijkstra_path_length(Calc_g, 0, rays * dots)
    way = nx.dijkstra_path(Calc_g, 0, rays * dots)
    
    for i in range(1, dots):
        buf_ans = nx.dijkstra_path_length(Calc_g, i, rays * dots + i)
        if buf_ans < ans:
            ans = buf_ans
            way = nx.dijkstra_path(Calc_g, i, rays * dots + i)
            
    way = np.array(way)
    return (ans, way)

#assigning number to certain pixel
def sector(dist, angle, rays, dots, radius):
    dist_k = float(dots) / radius
    dot_num = math.floor(dist * dist_k)
    dot_num = dots if dot_num >= dots else dot_num
    
    angle = angle + math.pi
    angle_k = float(rays) / (math.pi * 2.0)
    angle_num = math.floor(angle * angle_k)
    angle_num = rays if dot_num == dots else angle_num
    
    return np.array([dot_num, min(angle_num, rays - 1)])

#creating array containing in each cell a map describing one of the sectors 
#of the fragmentation map
def create_frag_map(radius, rays, dots):
    frag_map = np.zeros((rays * dots, 2 * radius + 1, 2 * radius + 1), dtype = bool)
    frag = np.zeros((2 * radius + 1, 2 * radius + 1, 2))
    
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            dist = np.sqrt(float((i - radius)**2 + (j - radius)**2))
            angle = np.angle(complex(i - radius, j - radius))
            frag[i, j] = sector(dist, angle, rays, dots, radius)
            
    for i in range(rays):
        for j in range(dots):
            segment_map = (frag == np.array([j, i]))
            segment_map = segment_map[:,:,0] * segment_map[:,:,1]
            frag_map[i * dots + j] = copy.deepcopy(segment_map)
            
            
    return frag_map

#single seeding point processing
def process_point(ext_im, ext_mask, frag_map, x_c, y_c, rays, dots, radius, lam):
    global Calc_g
    
    image_region = ext_im[x_c:x_c + 2 * radius + 1, y_c:y_c + 2 * radius + 1]
    mask_region = np.zeros([2 * radius + 1, 2 * radius + 1], dtype = np.uint16)
    
    for i in range(rays * dots):            
            mean_intensity = np.mean(image_region[frag_map[i]])                     
            weight = mean_intensity        
            
            if i % dots < dots - 1:
                Calc_g.edges[i, i + dots + 1]['weight'] = weight
            Calc_g.edges[i, i + dots]['weight'] = weight
            if i  % dots > 0:
                Calc_g.edges[i, i + dots - 1]['weight'] = weight
                
    (ans, way) = find_min_way(rays, dots)    
    mask_region = parse_way(mask_region, way, frag_map, rays)
    
    ext_mask[x_c:x_c + 2 * radius + 1, y_c:y_c + 2 * radius + 1] += mask_region
    return ext_mask

#main calculation graph initialization             
def initialize_graphs(rays, dots):
    global Calc_g
    
    H = nx.path_graph((rays + 1) * dots)
    Calc_g.add_nodes_from(H)
    
    for i in range(rays * dots):
            if i % dots < dots - 1:
                Calc_g.add_edge(i, i + dots + 1, weight = 1.0)
            Calc_g.add_edge(i, i + dots, weight = 1.0)
            if i % dots > 0:
                Calc_g.add_edge(i, i + dots - 1, weight = 1.0)

#finding sobel filter of image
def find_gaussian_gradient(image):
    gaussy = nd.gaussian_filter1d(image, 2, axis = 0, order = 1 )
    gaussy = nd.gaussian_filter1d(gaussy, 2, axis = 1, order = 0 )
    
    gaussx = nd.gaussian_filter1d(image, 2, axis = 1, order = 1 )
    gaussx = nd.gaussian_filter1d(gaussx, 2, axis = 0, order = 0 )
    
    sobel = np.arctan2(gaussx, gaussy) / np.pi
    return sobel

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
    erosed = nd.binary_erosion(image, np.ones((3,3)))
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
        angle = angles[bound[0], bound[1]]
        pixel = mask[bound[0], bound[1]]
        #смотрим на угол, проверяем необходимость удаления из маски     
        if ((angle <= -0.875) or (angle >= 0.875)) or ((angle >= -0.125) and (angle <= 0.125)):
            if max(mask[bound[0], bound[1] + 1], mask[bound[0], bound[1] - 1]) > pixel:
                mask[bound[0], bound[1]] = 0
        elif ((angle > -0.875) and (angle <= -0.625)) or ((angle > 0.125) and (angle <= 0.375)):
            if max(mask[bound[0] - 1, bound[1] - 1], mask[bound[0] + 1, bound[1] + 1]) > pixel:
                mask[bound[0], bound[1]] = 0
        if ((angle < 0.875) and (angle > 0.625)) or ((angle > -0.375) and (angle < -0.125)):
            if max(mask[bound[0] + 1, bound[1] - 1], mask[bound[0] - 1, bound[1] + 1]) > pixel:
                mask[bound[0], bound[1]] = 0
        else:
            if max(mask[bound[0] + 1, bound[1]], mask[bound[0] - 1, bound[1]]) > pixel:
                mask[bound[0], bound[1]] = 0
    
    return mask

def clear_mask(mask):
    st_bi_mask = np.where(mask[:,:] > 0, 0.0, 1.0)#начальная бинаризация    
    prev_label, prev_classes = nd.label(st_bi_mask)#начальная сегментация
    perim_mask = np.zeros(mask.shape)
    for j in range(15):
        bi_mask = np.where(mask[:,:] > 0, 1.0, 0.0)#получили границы
        angles = find_gaussian_gradient(bi_mask);#получили градиент на границах
        
        erosed = nd.binary_erosion(bi_mask, np.ones((3,3)))#провели эрозию
        erosed[0, :] = 1.0
        erosed[-1, :] = 1.0
        erosed[:, 0] = 1.0
        erosed[:, -1] = 1.0#исключили границы картинки
        perim = get_perimeter(bi_mask)#поулучили периметр 
        
        label, classes = nd.label(1.0 - erosed)#сегментация расширенной картинки
        if(classes != prev_classes):
             full_perim = np.argwhere((perim > 0)) #смотрим весь периметр 
             for pixel in full_perim:
                 if(perim_mask[pixel[0], pixel[1]] > 0):#если этот пиксель помечен в карте периметра не рассматриваем
                     continue
                 maxx = -np.Inf
                 minn = np.Inf
                 for m in range(-8,9): #просматриваем "окном" соотвествующего размера
                     for k in range(-8, 9):
                         buf = prev_label[min(prev_label.shape[0] - 1, max(0, pixel[0] + m)), 
                                          min(prev_label.shape[1] - 1, max(0, pixel[1] + k))]
                         if(buf > 0): 
                             maxx = max(maxx, buf)
                             minn = min(minn, buf)
                     #если встречаются две разные области в одном окне => проводим разделитель               
                     if (maxx != minn) and (minn != np.Inf):
                         perim_mask[pixel[0], pixel[1]] = 1.0
                         mask[pixel[0], pixel[1]] = 1.0
        print(j)
        cv.imwrite("../masks/mmask" + str(j) + ".png", mask*255 )#np.where((label > 0) * (get_perimeter(np.where(prev_label > 0, 0.0, 1.0))> 0) ,255.0, 0.0))
        
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
                     
                    
    cl_mask = nd.median_filter(nd.gaussian_filter(mask, 2), 5) * 10000 
    cl_mask = np.where(cl_mask < 128, 0,1)
    mask[mask > 0] = 1
    mask = skmorf.binary_opening(mask, init_dilate_kernel(5,5))
    mask[perim_mask > 0] = 1
    
    return mask #< 128, 1, 0))#nd.label(np.where(mask <  0.1, 1.0, 0.0))[0]
    
#clearing mask    
def clearr_mask(mask):
    st_bi_mask = np.where(mask[:,:] > 0, 0.0, 1.0)#начальная бинаризация    
    prev_label, prev_classes = nd.label(st_bi_mask)#начальная сегментация
    perim_mask = np.zeros(mask.shape)
    for j in range(15):
        bi_mask = np.where(mask[:,:] > 0, 1.0, 0.0)#получили границы
        angles = find_gaussian_gradient(bi_mask);#получили градиент на границах
        
        erosed = nd.binary_erosion(bi_mask, np.ones((3,3)))#провели эрозию
        erosed[0, :] = 1.0
        erosed[-1, :] = 1.0
        erosed[:, 0] = 1.0
        erosed[:, -1] = 1.0#исключили границы картинки
        perim = get_perimeter(bi_mask)#поулучили периметр 
        
        label, classes = nd.label(1.0 - erosed)#сегментация расширенной картинки
        if(classes != prev_classes):
             full_perim = np.argwhere((perim > 0)) #смотрим весь периметр 
             for pixel in full_perim:
                 if(perim_mask[pixel[0], pixel[1]] > 0):#если этот пиксель помечен в карте периметра не рассматриваем
                     continue
                 if pixel[0] == 0 or pixel[0] == label.shape[0] - 1 or pixel[1] == 0 or pixel[1] == label.shape[1] - 1:
                     continue
                 maxx = -np.Inf
                 minn = np.Inf
                 for m in range(-6,7): #просматриваем "окном" соотвествующего размера
                     for k in range(-6, 7):
                         buf = prev_label[min(prev_label.shape[0] - 1, max(0, pixel[0] + m)), 
                                          min(prev_label.shape[1] - 1, max(0, pixel[1] + k))]
                         if(buf > 0): 
                             maxx = max(maxx, buf)
                             minn = min(minn, buf)
                     #если встречаются две разные области в одном окне => проводим разделитель               
                     if (maxx != minn) and (minn != np.Inf):
                         perim_mask[pixel[0], pixel[1]] = 1.0
                         mask[pixel[0], pixel[1]] = 1.0
        print(j)
        cv.imwrite("../masks/mmask" + str(j) + ".png", mask*255 )#np.where((label > 0) * (get_perimeter(np.where(prev_label > 0, 0.0, 1.0))> 0) ,255.0, 0.0))
        
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
                     
                    
    cl_mask = nd.median_filter(nd.gaussian_filter(mask, 2), 5) * 10000 
    cl_mask = np.where(cl_mask < 128, 0,1)
    mask[mask > 0] = 1
    mask = skmorf.binary_opening(mask, init_dilate_kernel(5,5))
    mask[perim_mask > 0] = 1
    
    return mask #< 128, 1, 0))#nd.label(np.where(mask <  0.1, 1.0, 0.0))[0]

#making some basic improvements to mask
def mask_post_process(mask):    
    mask = np.double(mask)
    mask = mask / np.max(np.max(mask))
    
    dist_mask = np.where(mask < 0.3, 1, 0)
    distances = nd.distance_transform_edt(dist_mask)
        
    mask[distances > 30] = 0
    
    mask = nd.median_filter(mask, 3)
    mask[mask < 0.15] = 0
    mask[mask > 0.5] = 1
    
    
    cleared = clear_mask(copy.deepcopy(mask))
    
        
    return cleared
    
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
            
            
    predicted = ml.predict(ml.containerTestGenerator(samples), "D:\программы\Glands\Diploma\classic_gland_segmentation\ML\32B3L.hdf5", len(samples))
    for idx, pred in enumerate(predicted):
        if np.argmax(pred) == 1:
            label[label == idx + 1] = 0
    return label
    
#posttprocessing watershed
def watershed_post_process(mask):
    watershed = np.where(mask == 0, 1, 0)
    watershed = nd.label(watershed)[0]
    watershed = nd.watershed_ift(np.uint16(np.where(mask > 0, 1, 0)), watershed)
    #watershed = np.uint16(skmorf.watershed(mask, watershed, watershed_line = True))
    watershed = np.where(watershed > 1, 255, 0)
    
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
        if summ <= (0.10 - (0.07 if is_bordered else 0)) * maxx:
            watershed[watershed == i] = 0
            
    fig, axes = plt.subplots(ncols=2, figsize=(20, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(watershed * 255, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Изображение')
    plt.tight_layout()
    plt.show()  
    
    return watershed, classes

#post-processing bacis prediction results
def post_process(mask, image, filename):
    print("Post processing is already running..." )
    
    new_mask = mask_post_process(copy.deepcopy(mask))  
    mmask = copy.deepcopy(mask * new_mask)         
    watershed, classes = watershed_post_process(mmask)
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
    initialize_graphs(rays, dots)      
    mask = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint16)
    
    ext_im = np.pad(image, (radius, radius), 'reflect')
    ext_mask = np.pad(mask, (radius, radius), 'reflect')
    frag_map = create_frag_map(radius, rays, dots)
    
    
    for i in range(0, image.shape[0], 10):
        for j in range(0, image.shape[1], 10):
            ext_mask = process_point(ext_im, ext_mask, frag_map, i, j, rays, dots, radius, 0.0)
            
    mask = ext_mask[radius:-radius, radius:-radius]       
    
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