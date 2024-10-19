import numpy as np
from skimage import measure 
from typing import Union
from torch import Tensor
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import cv2

def get_rotated_bounding_box_parameters(trial_mask, rotation=False):
    '''
    When given a mask, this function will return the parameters of the minimum area
    rotated bounding box which can contain the mask.
    '''
    contours, _ = cv2.findContours(trial_mask.astype(np.uint8)*255, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    try:                
        all_points = np.concatenate([contour[:,0,:] for contour in contours])
        hull = cv2.convexHull(all_points)
    except:
        hull = contours[np.argmax([len(item) for item in contours])]

    if rotation:    
        ((center_x, center_y), (width, height), angle_of_rotation) = cv2.minAreaRect(hull)
        return ((center_x, center_y), (width, height), angle_of_rotation)
    else:
        x, y, w, h = cv2.boundingRect(hull)
        return ((x+w/2, y+h/2), (w, h), 0)

def create_bounding_box_mask(mask_shape, bounding_box_info):
    '''
    Given a mask shape and the bounding box parameters, 
    this function will return a mask with the bounding box filled
    '''
    box = cv2.boxPoints(bounding_box_info)
    box = np.intp(box)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 255, -1)
    return mask

def check_overlap_rotated(box1, box2):
    '''
    Given two rotated bounding boxes, this function will return the 
    percentage of of the second box with respect to the first box.
    (The larger box is always given as the first)
    '''
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    # Calculate intersection area
    intersection_area = poly1.intersection(poly2).area
    
    # Calculate area of the first bounding box
    box1_area = poly1.area
    
    # Check if intersection area is at least 50% of the first bounding box area
    return (intersection_area/box1_area)

## get the min distance between the centers - cdist
def get_min_distance_indices(coordinates):
    '''
    Given a list of coordinates, (the centers of each anomaly),
    this function will return the a list of the paris indices which correspond to the 
    centers which are closest to each other
    '''
    distances = cdist(coordinates, coordinates)

    np.fill_diagonal(distances, np.inf)

    # Find the indices of the closest coordinates
    closest_indices = np.argmin(distances, axis=1)
    min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    min_index
    
    out = []
    for a, b in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
        if a!=b:
            out.append((a, b))

    return np.array(out[::2])

def build_graph(pairs):
    graph = {}
    for pair in pairs:
        a, b = pair
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
    return graph

def dfs(graph, start, visited, component):
    visited.add(start)
    component.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, component)

def connected_components(pairs):
    graph = build_graph(pairs)
    visited = set()
    components = []
    for vertex in graph:
        if vertex not in visited:
            component = []
            dfs(graph, vertex, visited, component)
            components.append(component)
    return components

def sort_linked_sets(pairs):
    '''
    This takes a list of pairs, where each pair is a list of two indices,
    and those two indices represent the indices of the bounding boxes which are overlapping to each other.
    It returns a list of lists, where each internal list contains the indices of the bounding boxes which are overlapping with each other
    '''
    components = connected_components(pairs)
    return [sorted(component) for component in components]

def make_scaled_bb(temp_mask, min_target_scale, rotation):
    '''
    When given a mask, and the minimum target scale, this function will extract the rotated bounding box parameters,
    and if needed, scale the mask such that it meets the minimum target scale requirements
    '''
    ((center_x, center_y), (width, height), angle_of_rotation) = get_rotated_bounding_box_parameters(temp_mask, rotation)
    
    if width<256//min_target_scale:
        width = 256//min_target_scale

    if height<256//min_target_scale:
        height = 256//min_target_scale
        
    return ((center_x, center_y), (width, height), angle_of_rotation)

def get_overlapping_masks(bbs, overlap_limit=0.5):
    '''
    Given a list of rotated bounding box parameters, this returns a list of lists, 
    where each internal list contains the indices of the bounding boces which contain>overlap_limit
    with each other

    '''
    shape = (256, 256)
    bounding_box_infos = [item for item in bbs]
    
    coordinates = [item[0] for item in bounding_box_infos]
    dims = [item[1] for item in bounding_box_infos]
    out = get_min_distance_indices(coordinates)
    merge_list = []
    for indices in out:
        # box1, box2 = coordinates[indices[0]], coordinates[indices[1]]
        box1area = sum(dims[indices[0]])
        box2area = sum(dims[indices[1]])

        bounding_box_info1 = cv2.boxPoints(bounding_box_infos[indices[0]])
        bounding_box_info2 = cv2.boxPoints(bounding_box_infos[indices[1]])

        if box1area<box2area:
            # out = np.zeros(shape)
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2
            # out[out>0] = 255

            overlap = check_overlap_rotated(bounding_box_info1, bounding_box_info2)

        else:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            overlap = check_overlap_rotated(bounding_box_info2, bounding_box_info1)
        
        if overlap>overlap_limit:
            # out = np.zeros(shape)
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            # out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            # new_box = get_rotated_bounding_box_parameters(out)

            # new_bounding_box_infos = [item for i, item in enumerate(bounding_box_infos) if i not in [indices[0], indices[1]]]
            merge_list.append([indices[0], indices[1]])

            #restart = True
            #break
      
    groups = sort_linked_sets(merge_list)
    
    for indice in range(len(coordinates)):
        if not np.any([indice in group for group in groups]):
            groups.append([indice])
    return groups

def merge_overlapping(individual_masks, scaled_bbs, min_target_scale, rotation, overlap_limit=0.5):
    '''
    Given a list of individual masks and the corresponding bounding box parameters, the minimum target scale and the overlap limit,
    this will return a list of bounding box parameters which are the result of merging the masks which overlap with each other.
    We find two iterations of overlapping work best
    '''
    groups = get_overlapping_masks(scaled_bbs, overlap_limit=overlap_limit)
    out_masks = []
    for group in groups:
        out_mask = np.zeros((256, 256))
        for ind in group:
            out_mask += individual_masks[ind]
        out_masks.append(out_mask)
        
    scaled_bbs_2 = [make_scaled_bb(item, min_target_scale, rotation) for item in out_masks]
    
    groups = get_overlapping_masks(scaled_bbs_2, overlap_limit=overlap_limit)
    out_masks_2 = []
    for group in groups:
        out_mask = np.zeros((256, 256))
        for ind in group:
            out_mask += out_masks[ind]
        out_masks_2.append(out_mask)
    scaled_bbs_3 = [make_scaled_bb(item, min_target_scale, rotation) for item in out_masks_2]
    
    return scaled_bbs_3

import torch
def target_to_rotated_scaled_merged_BBs(target, overlap_limit, min_target_scale, rotation=False, shape=(256, 256)):
    '''
    Given a target mask (which may contain many non-overlapping anomalies), this function will return a list of bounding box parameters
    of each individual anomaly, given the overlap_limit, and the minimum target scale 
    returns masks in the format cxcywh
    '''
    if target.max()==0:
        return torch.empty(0, 4)

    regions = [region for region in measure.regionprops(measure.label(target>0))]
    
    if len(regions)==1:
        bbs = [make_scaled_bb(target, min_target_scale, rotation)]
        if rotation:
            bbs = torch.tensor([(item[0][0], item[0][1], item[1][0], item[1][1], item[2]) for item in bbs])
        else:
            bbs = torch.tensor([(item[0][0], item[0][1], item[1][0], item[1][1]) for item in bbs])
        return bbs
    
    scaled_bbs = []
    individual_masks = []
    for region_ind, region in enumerate(regions):
        temp_mask = np.zeros((256, 256))
        temp_mask[tuple([[item[0] for item in region.coords], [item[1] for item in region.coords]])] = 1
        individual_masks.append(temp_mask)
        scaled_bbs.append(make_scaled_bb(temp_mask, min_target_scale, rotation))
    
    bbs = merge_overlapping(individual_masks, scaled_bbs, min_target_scale, rotation, overlap_limit=overlap_limit)
    if rotation:
        bbs = torch.tensor([(item[0][0], item[0][1], item[1][0], item[1][1], item[2]) for item in bbs])
    else:
        bbs = torch.tensor([(item[0][0], item[0][1], item[1][0], item[1][1]) for item in bbs])
    return bbs


def convert_cxcywh_to_bbox(cx, cy, w, h, theta_degrees=0):
    """
    Converts a bounding box from (cx, cy, w, h, theta) format to 
    (x1, y1, x2, y2, x3, y3, x4, y4) format, with the angle in degrees.
    
    Parameters:
    cx, cy - center of the bounding box
    w, h - width and height of the bounding box
    theta_degrees - rotation angle in degrees (default is 0, which means no rotation)
    
    Returns:
    A list of 8 elements representing the four corners of the bounding box
    in the order (x1, y1, x2, y2, x3, y3, x4, y4).
    """
    
    # Convert the angle from degrees to radians
    theta = np.radians(theta_degrees)
    
    # Half dimensions
    w_half = w / 2
    h_half = h / 2
    
    # Define the corners relative to the center before rotation
    corners = np.array([
        [-w_half, -h_half],  # Top-left
        [w_half, -h_half],   # Top-right
        [w_half, h_half],    # Bottom-right
        [-w_half, h_half]    # Bottom-left
    ])
    
    # Rotation matrix for the angle theta
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate the corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate the corners to the center (cx, cy)
    rotated_corners += np.array([cx, cy])
    
    # Flatten the rotated corners to a 1D list (x1, y1, x2, y2, x3, y3, x4, y4)
    flattened_corners = rotated_corners.flatten()
    
    return flattened_corners.tolist()



def convert_bbox_to_cxcywh(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Converts a bounding box from (x1, y1, x2, y2, x3, y3, x4, y4) format 
    to (cx, cy, w, h, theta) format, where theta is in degrees.
    
    Parameters:
    x1, y1, x2, y2, x3, y3, x4, y4 - The four corners of the bounding box
    
    Returns:
    cx, cy, w, h, theta - Center, width, height, and rotation angle in degrees
    """
    
    # Calculate the center (cx, cy)
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4
    
    # Calculate the width as the distance between top-left and top-right corners
    w = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Calculate the height as the distance between top-left and bottom-left corners
    h = np.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)
    
    # Calculate the rotation angle theta (in radians first) using the angle between (x1, y1) and (x2, y2)
    theta_radians = np.arctan2(y2 - y1, x2 - x1)
    
    # Convert the angle to degrees
    theta_degrees = np.degrees(theta_radians)
    
    return cx, cy, w, h, theta_degrees

def cxcywh_to_xxyyxxyy(bbs, rev=False):
    if not rev:
        return torch.tensor([convert_cxcywh_to_bbox(*bb) for bb in bbs])
    else:
        return torch.tensor([convert_bbox_to_cxcywh(*bb) for bb in bbs])
    

def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)

def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)