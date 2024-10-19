import os
import numpy as np
from skimage import measure
from skimage.transform import resize

def downscale_mask(mask, downscale_factor = 2):
    original_image = mask

    # Downsample the image
    downsampled_image = resize(mask>0, (mask.shape[0] // downscale_factor, mask.shape[1] // downscale_factor),
                               anti_aliasing=False)
    
    # Upsample the image
    scaled_image = resize(downsampled_image, mask.shape, order=0, anti_aliasing=False)
    
    return (scaled_image > 0.5).astype(int)

def extract_bb_coords_mask(mask):
    mask = np.array(mask)
    
    mask = downscale_mask(mask, 3)
    
    if mask.shape[0]==1:
        _, height, width = mask.shape
    elif mask.shape[-1]==1:
        height, width, _ = mask.shape
    else:
        height, width = mask.shape
    
    regions = [region for region in measure.regionprops(measure.label(mask))]
    out = []
    for region in regions:
        y_min = region.coords[:,0].min()
        y_max = region.coords[:,0].max()
        x_min = region.coords[:,1].min()
        x_max = region.coords[:,1].max()
        
        if x_min<0:
            x_min = 0
        if y_min<0:
            y_min = 0
        
        if x_max>width:
            x_max = width
        if y_min>height:
            y_min = height
            
        bb_width  = x_max-x_min
        bb_height = y_max-y_min
        
        if bb_width<1:
            bb_width = 2
        if bb_height<1:
            bb_height = 2
        
        center_x = (x_min+x_max)/2/width
        center_y = (y_min+y_max)/2/height
        
        if center_x>1:
            center_x=1
        if center_y>1:
            center_y=1
        if center_x<0:
            center_x=0
        if center_y<0:
            center_y=0
            
        out.append([center_x,
                    center_y,
                    bb_width/mask.shape[1],
                    bb_height/mask.shape[0],
                   ]
                  )
    return out

def create_dataset(dataset_key, unique_key = ""):
    dataset_dir = os.path.join(os.getcwd(), "yolo_datasets")
    string = '''
    # YOLOCore
    # Amended code from Ultralytics YOLO, AGPL-3.0 license
    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: __dataset_dir__/__dataset__ # dataset root dir
    train: images/train # train images (relative to 'path') 128 images
    val: images/test # val images (relative to 'path') 128 images
    test: # test images (optional)

    # Classes
    names:
      0: anomaly
    '''.replace("__dataset__", dataset_key+"_"+unique_key).replace("__dataset_dir__", dataset_dir)
    
    os.makedirs(os.path.join(os.getcwd(), "ultralytics", "cfg", "datasets"), exist_ok=True)
    cfg_path = os.path.join(os.getcwd(), "ultralytics", "cfg", "datasets", dataset_key+"_"+unique_key+".yaml")
    with open(cfg_path, "w") as f:
        f.write(string)
        
    return cfg_path, dataset_dir

def min_max_scale(arr, arr_min, arr_max, min_val=0.2, max_val=0.8):
    return min_val + (arr - arr_min) * (max_val - min_val) / (arr_max - arr_min)

def z_norm(arr, arr_mean, arr_std, center=0.5, width=0.15):
    return ((arr-arr_mean)/arr_std)*width+center
    #return min_val + (arr - arr_min) * (max_val - min_val) / (arr_max - arr_min)
