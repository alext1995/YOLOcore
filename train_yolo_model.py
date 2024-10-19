import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage import measure
from PIL import Image
import os
import pandas as pd
import json
from ultralytics.models.yolo.detect import DetectionValidator
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from ultralytics.models.yolo.obb import OBBValidator

def calculate_area(box):
    # Box is [x_min, y_min, x_max, y_max]
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width * height

def calculate_intersection(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    # Calculate intersection area
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    
    return width * height

def intersection_over_area_of_smallest_box(box1, box2):
    # Calculate the intersection area
    intersection_area = calculate_intersection(box1, box2)
    
    # Calculate the area of both boxes
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    
    # Find the smallest area
    smallest_area = min(area1, area2)
    
    # If there is no intersection, return 0
    if intersection_area == 0 or smallest_area == 0:
        return 0
    
    # IoA: Intersection over the area of the smallest box
    return intersection_area / smallest_area

def filter_bounding_boxes_recursive(bboxes, threshold):
    """
    Recursively filters bounding boxes by removing any boxes with an IoA 
    less than a given threshold.

    :param bboxes: List of bounding boxes in the format [(x1, y1, x2, y2), ...]
    :param threshold: The IoA threshold to decide if a box should be removed
    :return: A filtered list of bounding boxes
    """
    
    # Base case: if the list is empty or has only one bounding box, return it as is
    if len(bboxes) <= 1:
        return bboxes
    
    # Take the first bounding box from the list
    first_box = bboxes[0]
    filtered_boxes = [first_box]  # Start with the first box in the filtered list

    # Iterate over the rest of the bounding boxes
    remaining_boxes = bboxes[1:]

    # Recursively check each of the remaining boxes
    def should_keep_box(box):
        # Check IoA between the current box and all previously filtered boxes
        for filtered_box in filtered_boxes:
            if intersection_over_area_of_smallest_box(filtered_box, box) >= threshold:
                return False
        return True

    # Keep only the boxes that meet the IoA threshold
    remaining_filtered_boxes = [box for box in remaining_boxes if should_keep_box(box)]

    # Recursively process the remaining boxes
    return filtered_boxes + filter_bounding_boxes_recursive(remaining_filtered_boxes, threshold)

def filter_bounding_boxes_ioa(boxes, threshold=0.9):
    if len(boxes) == 0:
        return torch.empty(0, 5)
    bounding_boxes_out = filter_bounding_boxes_recursive(boxes, threshold=threshold)
    bounding_boxes_out = [np.array(item) for item in bounding_boxes_out]
    if isinstance(bounding_boxes_out, list):
        return torch.tensor(np.stack(bounding_boxes_out))
    return torch.tensor(bounding_boxes_out)  
    
def get_labels(label_path):
    if os.path.getsize(label_path)==0:
        return torch.empty(0, 4)
    return torch.tensor([[float(i) for i in item.split(" ")] for item in pd.read_csv(label_path, header=None, index_col=False).values[:,0]])[:,1:]

def load_im2im_paths(cfg):
    if cfg[-5:]==".yaml":
        cfg = cfg[:-5]
    path = os.path.join(os.getcwd(), 
                "ultralytics", 
                "cfg", 
                "datasets", 
                cfg+"_im2im_paths.json")
    with open(path, "r") as f:
        return json.load(f)
    
from torchvision.ops import box_convert

def calculate_area(box):
    # Box is [x_min, y_min, x_max, y_max]
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width * height

def calculate_intersection(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    # Calculate intersection area
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    
    return width * height

def intersection_over_area_of_smallest_box(box1, box2):
    # Calculate the intersection area
    intersection_area = calculate_intersection(box1, box2)
    
    # Calculate the area of both boxes
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    
    # Find the smallest area
    smallest_area = min(area1, area2)
    
    # If there is no intersection, return 0
    if intersection_area == 0 or smallest_area == 0:
        return 0
    
    # IoA: Intersection over the area of the smallest box
    return intersection_area / smallest_area

def filter_bounding_boxes_recursive(bboxes, threshold):
    """
    Recursively filters bounding boxes by removing any boxes with an IoA 
    less than a given threshold.

    :param bboxes: List of bounding boxes in the format [(x1, y1, x2, y2), ...]
    :param threshold: The IoA threshold to decide if a box should be removed
    :return: A filtered list of bounding boxes
    """
    
    # Base case: if the list is empty or has only one bounding box, return it as is
    if len(bboxes) <= 1:
        return bboxes
    
    # Take the first bounding box from the list
    first_box = bboxes[0]
    filtered_boxes = [first_box]  # Start with the first box in the filtered list

    # Iterate over the rest of the bounding boxes
    remaining_boxes = bboxes[1:]

    # Recursively check each of the remaining boxes
    def should_keep_box(box):
        # Check IoA between the current box and all previously filtered boxes
        for filtered_box in filtered_boxes:
            if intersection_over_area_of_smallest_box(filtered_box, box) >= threshold:
                return False
        return True

    # Keep only the boxes that meet the IoA threshold
    remaining_filtered_boxes = [box for box in remaining_boxes if should_keep_box(box)]

    # Recursively process the remaining boxes
    return filtered_boxes + filter_bounding_boxes_recursive(remaining_filtered_boxes, threshold)

def filter_bounding_boxes_ioa(boxes, threshold=0.9):
    if len(boxes) == 0:
        return torch.empty(0, 5)
    bounding_boxes_out = filter_bounding_boxes_recursive(boxes, threshold=threshold)
    bounding_boxes_out = [np.array(item) for item in bounding_boxes_out]
    if isinstance(bounding_boxes_out, list):
        return torch.tensor(np.stack(bounding_boxes_out))
    return torch.tensor(bounding_boxes_out)  
    
def get_labels(label_path):
    if os.path.getsize(label_path)==0:
        return torch.empty(0, 4)
    return torch.tensor([[float(i) for i in item.split(" ")] for item in pd.read_csv(label_path, header=None, index_col=False).values[:,0]])[:,1:]

def load_im2im_paths(cfg):
    if cfg[-5:]==".yaml":
        cfg = cfg[:-5]
    path = os.path.join(os.getcwd(), 
                "ultralytics", 
                "cfg", 
                "datasets", 
                cfg+"_im2im_paths.json")
    with open(path, "r") as f:
        return json.load(f)
    
def make_3d(image_2d):
    return torch.stack([image_2d]*3)

from ultralytics import YOLO

def main(name_key, result_key, device, args, rotation=False):
    results_dir = "results"
    os.makedirs(os.path.join(results_dir, name_key, result_key), exist_ok=True)

    # Load a model

    if rotation:
        model = args.model+"-obb"
    else:
        model = args.model

    model = YOLO(f"{model}.pt")

    # Train the model
    train_results = model.train(
    #     data="mvtec_bottle_test_2.yaml",  # path to dataset YAML
        data=name_key+".yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=256,  # training image size
        device=int(device.split(":")[1]),  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        single_cls=True,
    )
    stats = train_results.results_dict

    if not rotation:
        val_class = DetectionValidator
    else:
        val_class = OBBValidator

    val = val_class()
    val.iouv[:] = 0.25
    model.trainer.validator.metrics = val.metrics
    model.trainer.validator.iouv[:] = 0.25
    stats_25 = model.trainer.validator(model.trainer)

    stats['metrics/precision(B)25'] =  stats_25['metrics/precision(B)']
    stats['metrics/recall(B)25']    =  stats_25['metrics/recall(B)']
    stats['metrics/mAP25(B)']       =  stats_25['metrics/mAP50(B)']
    with open(os.path.join(results_dir, name_key, result_key, "ultralytics_stats.json"), "w") as f:
        json.dump(stats, f) 
    
    results = model(model.trainer.get_dataset()[1], iou=0.9)  # return a list of Results objects

    boxes_list = []
    masks_list = []
    confs_list = []
    path_list = []

    for result in results:
        if not rotation:
            boxes = json.dumps(result.boxes.xyxy.cpu().numpy().tolist())  # Boxes object for bounding box outputs
            confs = json.dumps(result.boxes.conf.cpu().numpy().tolist())  # Probs object for classification outputs
        else:
            boxes = json.dumps(result.obb.xywhr.cpu().numpy().tolist())  # Boxes object for bounding box outputs
            confs = json.dumps(result.obb.conf.cpu().numpy().tolist())  # Probs object for classification outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        boxes_list.append(boxes)
        masks_list.append(masks)
        path_list.append(result.path)
        confs_list.append(confs)
        
    import pandas as pd
    df = pd.DataFrame({"boxes": boxes_list, 
                        "masks": masks_list, 
                        "confs": confs_list, 
                        "paths": path_list})
    im2im_paths = load_im2im_paths(os.path.split(model.trainer.args.data)[1])

    df.to_csv(os.path.join(results_dir, name_key, result_key, "results_df.csv"))

    if rotation:
        return 
    
    for demo_batch_item in model.trainer.validator.dataloader:
        break

    demo_img_shape = demo_batch_item["img"].shape
    ratio_pad = demo_batch_item["ratio_pad"][0]

    val = DetectionValidator()
    val.seen = 0
    val.device = model.device
    val.data = model.trainer.data
    val.init_metrics(model)

    val_25 = DetectionValidator()
    val_25.iouv = torch.linspace(0.25, 0.25, 10)
    val_25.seen = 0
    val_25.device = model.device
    val_25.data = model.trainer.data
    val_25.init_metrics(model)

    t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    for ii, (boxes, confs, path) in enumerate(df[["boxes", "confs", "paths"]].values[:, :]):
        
        boxes = json.loads(boxes)
        confs = json.loads(confs)

        if len(confs)==0:
            pass
            
        boxes_with_confs = torch.hstack((torch.tensor(boxes), 
                                         torch.tensor(confs)[:,None])) 
        filtered_boxes = filter_bounding_boxes_ioa(boxes_with_confs, 
                                                   threshold=0.8)
            
        '''
        # plotting stuff

        image = t(Image.open(path))

        drawn_image = (make_3d(image[0])*255).to(torch.uint8)
        if len(filtered_boxes)>0:
            drawn_image = draw_bounding_boxes(drawn_image, 
                                            filtered_boxes[:,:4],
                                            colors ="red",
                                            width  = 2)
        '''

        original, intermediate, label_path = im2im_paths[os.path.split(path)[1]]
        
        label_box_xywhn = get_labels(label_path)
        
        '''
        # plotting stuff

        label_box_xyxy = 256*box_convert(label_box_xywhn, 
                                        in_fmt="cxcywh", 
                                        out_fmt="xyxy")
        
        if len(label_box_xyxy)>0:
            drawn_image = draw_bounding_boxes(drawn_image, 
                                            label_box_xyxy,
                                            colors ="blue",
                                            width  = 2)

        final_image = transforms.ToPILImage()(drawn_image)
        
        original = transforms.ToPILImage()(t(Image.open(original)))
        
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original)
        axes[1].imshow(final_image)
        '''
        len_batch = 1
        batch = {"batch_idx": torch.zeros(label_box_xywhn.shape[0]).to(model.device),
                "cls": torch.zeros(label_box_xywhn.shape[0], 1).to(model.device),
        #          "bboxes": torch.stack([label_box_xywhn]).to(model.device),
                "bboxes": label_box_xywhn.to(model.device),
                "ori_shape": [(256, 256)]*len_batch,
                "imsz": torch.empty(demo_img_shape),
                "img": torch.empty(len_batch, 1, 256, 256).to(model.device),
                "ratio_pad": [ratio_pad]*len_batch}
        pred = torch.hstack((filtered_boxes, 
                            torch.zeros(filtered_boxes.shape[0], 1))).to(model.device)
        val.update_metrics([pred], batch)
        val.run_callbacks("on_val_batch_end")
    
        val_25.update_metrics([pred], batch)
        val_25.run_callbacks("on_val_batch_end")
        
    stats = val.get_stats()
    stats_25 = val_25.get_stats()

    stats['metrics/precision(B)25'] =  stats_25['metrics/precision(B)']
    stats['metrics/recall(B)25']    =  stats_25['metrics/recall(B)']
    stats['metrics/mAP25(B)']       =  stats_25['metrics/mAP50(B)']

    val.check_stats(stats)
    val.finalize_metrics()
    val.print_results()

    with open(os.path.join(results_dir, name_key, result_key, "custom_stats.json"), "w") as f:
        json.dump(stats, f) 

import sys

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Compute AP50 and mAP (AP50:95) for YOLOv8 model predictions.")

    # Add arguments
    parser.add_argument('--dataset_key', type=str, required=True, help='Dataset name')
    parser.add_argument('--ad_device', type=str, required=True, help='GPU identifier - etc "cuda:0"')
    parser.add_argument('--unique_key', type=str, required=True, help='unique_key')
    parser.add_argument('--results_key', type=str, default="standard", required=False, help='directly name that these results go into')
    parser.add_argument('--model', type=str, default='yolov8n', required=False, help='model key')
    
    args = parser.parse_args()

    name_key = args.dataset_key+"_"+args.unique_key
    result_key = args.results_key
    device = args.ad_device
    if name_key[-3:]=="rot":
        rotation = True
    else:
        rotation= False
    print("Rotation: ", rotation)
    main(name_key, result_key, device, args, rotation)

