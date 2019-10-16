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


Graph = None
Calc_g = nx.DiGraph()

def parse_way(mask_region, way, frag_map):
    global Graph
    sh = Graph.shape
    for i in range(sh[0]):
        gr = np.uint16(Graph[i, way[i]])
        
        segment_map = (frag_map == gr)
        segment_map = segment_map[:,:,0] * segment_map[:,:,1]

        mask_region[segment_map] += 1
    return mask_region
        
    
def find_min_way(rays, dots):
    global Calc_g
    ans = nx.dijkstra_path_length(Calc_g, 0, rays * dots)
    way = nx.dijkstra_path(Calc_g, 0, rays * dots)
    for i in range(1, dots):
        buf_ans = nx.dijkstra_path_length(Calc_g, i, rays * dots + i)
        if buf_ans < ans:
            ans = buf_ans
            way = nx.dijkstra_path(Calc_g, i, rays * dots + i)
    way = np.array(way) % dots
    return (ans, way)

def sector(dist, angle, rays, dots, radius):
    dist_k = float(dots) / radius
    dot_num = math.floor(dist * dist_k)
    dot_num = dots if dot_num >= dots else dot_num
    
    angle = angle + math.pi
    angle_k = float(rays) / (math.pi * 2.0)
    angle_num = math.floor(angle * angle_k)
    angle_num = rays if dot_num == dots else angle_num
    
    return np.array([dot_num, min(angle_num, rays - 1)])

def create_frag_map(radius, rays, dots):
    frag = np.zeros((2 * radius + 1, 2 * radius + 1, 2))
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            dist = np.sqrt(float((i - radius)**2 + (j - radius)**2))
            angle = np.angle(complex(i - radius, j - radius))
            frag[i, j] = sector(dist, angle, rays, dots, radius)
    return frag

def process_point(ext_im, ext_mask, frag_map, x_c, y_c, rays, dots, radius, lam):
    global Graph
    global Calc_g
    
    image_region = ext_im[x_c:x_c + 2 * radius + 1, y_c:y_c + 2 * radius + 1]
#    sobel_region = ext_sobel[x_c:x_c + 2 * radius + 1, y_c:y_c + 2 * radius + 1]
    mask_region = np.zeros([2 * radius + 1, 2 * radius + 1], dtype = np.uint16)
    
    for j in range(rays):
#        summ = 0.0
        for i in range(dots):
            segment_map = (frag_map == np.array([i, j]))
            segment_map = segment_map[:,:,0] * segment_map[:,:,1]
            
            mean_intensity = np.mean(image_region[segment_map])
#            mean_sobel = np.mean(sobel_region[segment_map])
            
#            summ += mean_sobel           
            weight = mean_intensity #+ summ * lam
            
            
            Graph[j, i] = [i, j]
            
            if i < dots - 1:
                Calc_g.edges[j * dots + i, (j + 1) * dots + i + 1]['weight'] = weight
            Calc_g.edges[j * dots + i, (j + 1) * dots + i]['weight'] = weight
            if i > 0:
                Calc_g.edges[j * dots + i, (j + 1) * dots + i - 1]['weight'] = weight
                
    (ans, way) = find_min_way(rays, dots)    
    mask_region = parse_way(mask_region, way, frag_map)
    
    ext_mask[x_c:x_c + 2 * radius + 1, y_c:y_c + 2 * radius + 1] += mask_region
    return ext_mask

            
def initialize_graphs(rays, dots):
    global Calc_g
    global Graph
    Graph = np.zeros([rays, dots, 2])
    H = nx.path_graph((rays + 1) * dots)
    Calc_g.add_nodes_from(H)
    for j in range(rays):
        for i in range(dots):
            if i < dots - 1:
                Calc_g.add_edge(j * dots + i, (j + 1)* dots + i + 1, weight = 1.0)
            Calc_g.add_edge(j * dots + i, (j + 1)* dots + i, weight = 1.0)
            if i > 0:
                Calc_g.add_edge(j * dots + i, (j + 1)* dots + i - 1, weight = 1.0)

def find_sobel(image):
    sobelx = np.array(cv.Sobel(image, 3, 1, 0), dtype = np.float)
    sobely = np.array(cv.Sobel(image, 3, 0, 1), dtype = np.float)
    
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    return sobel

def init_dilate_kernel(size_x, size_y):
    dilate_kernel = np.ones((size_x,size_y), dtype = np.uint8)
    
    dilate_kernel[0, -1] = 0
    dilate_kernel[-1, -1] = 0
    dilate_kernel[0, 0] = 0
    dilate_kernel[-1, 0] = 0
    
    return dilate_kernel
        

def make_sp(image, rays, dots, radius):    
    fig, axes = plt.subplots(ncols=2, figsize=(20, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    
    initialize_graphs(rays, dots)      
    mask = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint16)
    
    ext_im = np.pad(image, (radius, radius), 'reflect')
    ext_mask = np.pad(mask, (radius, radius), 'reflect')
    frag_map = create_frag_map(radius, rays, dots)
    
    for i in range(0, image.shape[0], 10):
        for j in range(0, image.shape[1], 10):
            ext_mask = process_point(ext_im, ext_mask, frag_map, i, j, rays, dots, radius, 0.0)
#            print("OK", j, i)
#    ext_mask = process_point(ext_im, ext_mask, ext_sobel, frag_map, 70, 70, rays, dots, radius, 0.0)
    mask = ext_mask[radius:-radius, radius:-radius]
    
    kernel = np.ones((3,3))
    dilate_kernel = init_dilate_kernel(21,21)
    smooth_kernel = init_dilate_kernel(7,7)
    
    mask = np.double(mask)
    mask = mask / np.max(np.max(mask))
    mask[mask < 0.15] = 0
    mask = np.uint8(mask * 255)
    
    ax[0].imshow(mask, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Markers')
    
    
#    mask3 = nd.grey_closing(mask3, (5,5))
    watershed = np.where(mask[:,:] < 0.1, 1, 0)
    watershed = nd.label(watershed)[0]
    watershed = np.uint16(nd.watershed_ift(mask, watershed))
#    ax[0].imshow(watershed, cmap=plt.cm.gray, interpolation='nearest')
#    ax[0].set_title('Markers')
#    
    maxx = 0
    
    watershed[watershed > 0] = 1
#    watershed = nd.binary_dilation(watershed, structure = dilate_kernel)
    watershed, classes = nd.label(watershed)
    
    for i in range(1, classes):
        summ = sum(sum(np.where(watershed == i, 1, 0)))
        if summ > maxx:
            maxx = summ;
            
    for i in range(1, classes):
        summ = sum(sum(np.where(watershed == i, 1, 0)))
        if summ <= 0.3 * maxx:
            watershed[watershed == i] = 0
            
    watershed[watershed > 0] = 1        
    watershed = nd.binary_dilation(watershed, structure = dilate_kernel)
    
    watershed, classes = nd.label(watershed)
    
    for i in range(1, classes):
        segment_map = np.where(watershed == i, 1, 0)
        summ = sum(sum(segment_map))
        perim = skm.perimeter(segment_map, neighbourhood=8)
        
        if (float(perim) / np.sqrt(summ)) > 12   :
            watershed[watershed == i] = 0
    
    
    new_im = cv.cvtColor(np.uint8(image), cv.COLOR_GRAY2BGR)
    
    new_im[watershed > 0, 1:2] = 0
    
#    cv.imwrite("./wtshd.png", watershed)
          
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
    images.append(data)

if not images[0] is None:
   i = 1
   for img in images:
       start_time = time.time()
#       fig, axes = plt.subplots(ncols=2, figsize=(20, 8), sharex=True, sharey=True)
#       ax = axes.ravel()
       print(img.shape)
       
       img = make_sp(img, 10, 8, 120)
       cv.imwrite("../res1/" + str(i) + ".png" , img)
       i = i + 1
       
#       ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
#       ax[0].set_title('Markers')
#       for a in ax:
#           a.set_axis_off()
#        
#       fig.tight_layout()
#       plt.show()
       print(time.time() - start_time)