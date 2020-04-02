import numpy as np
import networkx as nx
import math
import copy


STEP = 10
LAMBDA = 0.0

#parsing the way and adding changes to mask
def parse_way(mask_region, way, frag_map, rays):
    for i in range(rays):
        mask_region[frag_map[way[i]]] += 1
    return mask_region
        
#function that findes way with mimimal sum of node weights 
def find_min_way(Calc_g, rays, dots):
    ans = nx.dijkstra_path_length(Calc_g, 0, rays * dots)
    way = nx.dijkstra_path(Calc_g, 0, rays * dots)
    
    for i in range(1, dots):
        buf_ans = nx.dijkstra_path_length(Calc_g, i, rays * dots + i)
        if buf_ans < ans:
            ans = buf_ans
            way = nx.dijkstra_path(Calc_g, i, rays * dots + i)
            
    way = np.array(way)
    return (ans, way), Calc_g

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
def process_point(Calc_g, ext_im, ext_mask, frag_map, x_c, y_c, rays, dots, radius, lam):    
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
                
    (ans, way), Calc_g = find_min_way(Calc_g, rays, dots)    
    mask_region = parse_way(mask_region, way, frag_map, rays)
    
    ext_mask[x_c:x_c + 2 * radius + 1, y_c:y_c + 2 * radius + 1] += mask_region
    return ext_mask, Calc_g

#main calculation graph initialization             
def initialize_graphs(rays, dots):
    Calc_g = nx.DiGraph()
    
    H = nx.path_graph((rays + 1) * dots)
    Calc_g.add_nodes_from(H)
    
    for i in range(rays * dots):
            if i % dots < dots - 1:
                Calc_g.add_edge(i, i + dots + 1, weight = 1.0)
            Calc_g.add_edge(i, i + dots, weight = 1.0)
            if i % dots > 0:
                Calc_g.add_edge(i, i + dots - 1, weight = 1.0)
    return Calc_g

def find_bound_mask(image, rays, dots, radius):
    Calc_g = initialize_graphs(rays, dots)      
    mask = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint16)
    
    ext_im = np.pad(image, (radius, radius), 'reflect')
    ext_mask = np.pad(mask, (radius, radius), 'reflect')
    frag_map = create_frag_map(radius, rays, dots)
    
    
    for i in range(0, image.shape[0], STEP):
        for j in range(0, image.shape[1], STEP):
            ext_mask, Calc_g = process_point(Calc_g, ext_im, ext_mask, 
                                             frag_map, i, j, rays, dots, 
                                             radius, LAMBDA)
            
    mask = ext_mask[radius:-radius, radius:-radius]
    return mask

if __name__ == "__main__":
    print("This module can't be executed independently")
