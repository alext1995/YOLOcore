from synthetic_anomalies.synthetic_anomalies import StratifiedMaskHolder
from visionad_wrapper.data.data_loading import get_dataloaders
import pickle
from patchcoreplus.patchcoreplus import WrapperPatchCorePlus
import sys
import torch
import os
from torchvision.ops import box_convert
from yolo_integrator.creating_yolo_datasets import min_max_scale, create_dataset, downscale_mask, z_norm
from PIL import Image
import numpy as np
import json
from visionad_wrapper.wrapper.wrapper import run_metrics
from yolo_integrator.filtering_bounding_box_labels import target_to_rotated_scaled_merged_BBs, xywhr2xyxyxyxy
from train_yolo_model import main as train_yolo_main

'''
This file creates training and testing files for a given Patchcore model and the synthetic anomaly parameters.
The output files are in the COCO dataset format. 
'''

pc_hyperparameters = {"results_path":"test_results", #, type=str)
                        "seed":0, #, ,type=int, default=0, show_default=True)
                        "log_group":"group", #, type=str, default="group")
                        "log_project":"project", #, type=str, default="project")
                        "backbone_names_layers": {"wideresnet50": ["layer2", "layer3"]}, # each item must be a list of layers
                        "pretrain_embed_dimension":1024, #, type=int, default=1024)
                        "target_embed_dimension":1024, #, type=int, default=1024)
                        "preprocessing":"mean", #, type=click.Choice(["mean", "conv"]), default="mean")
                        "aggregation":"mean", #, type=click.Choice(["mean", "mlp"]), default="mean")
                        "anomaly_scorer_num_nn":5,#, ,type=int, default=5)
                        "patchscore":"max", # , type=str, default="max")
                        "patchoverlap":0.0 ,#, type=float, default=0.0)
                        "patchsize_aggregate":[] ,#, "-pa", type=int, multiple=True, default=[])
                        "faiss_on_gpu": True,#, is_flag=True)
                        "faiss_num_workers":8, #, ,type=int, default=8)
                        "num_workers":8, #, ,default=8, type=int, show_default=True)
                        "input_size": 256,#, default=256, type=int, show_default=True)
                        "augment":True,#, is_flag=True)
                        "coreset_size": 25000, #, "-p", type=float, default=0.1, show_default=True)
                        "coreset_search_dimension_map": 128,
                        "nn_code": "new",
                        "distance_type": "euclidean",
                        "patchstride": 1,
                        "patchsize": 3,
}


class SyntheticTrainingDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_holder = {}
        
        self.images = []
        self.paths  = []
        self.masks  = []
        self.to_pass = []
        for _, items in enumerate(self.dataloader):
            image_batch, paths_batch, callback_info_batch = items
            self.images.append(image_batch)       
            self.paths.append(paths_batch)         
            self.masks.append(callback_info_batch)
            self.to_pass.append((image_batch, callback_info_batch,))
            
    def __len__(self,):
        return len(self.to_pass)

def make_synthetic_training_ad_dataloader(dataloader_train, 
                                          mask_holder_params,
                                          amount=0):
    stratmask = StratifiedMaskHolder(**mask_holder_params)
    dataloader_train.dataset.pre_normalisation_transform = stratmask.add_noise_to_image 
    
    with torch.no_grad():
        if amount==0:
            datas_train = [items for _, items in enumerate(dataloader_train)]
        else:
            datas_train = []
            while len(datas_train)<amount//dataloader_train.batchsize:
                datas_train += [items for _, items in enumerate(dataloader_train)]
            datas_train = datas_train[:(amount//dataloader_train.batchsize)]
            
    return SyntheticTrainingDataLoader(datas_train)

def main(dataset_key, ad_device, unique_key, args):
    dataloader_train, dataloader_regular_test, dataloader_novel_test = get_dataloaders(dataset_key, ad_device)

    mask_params = {"weight_min": args.weight_min,
                    "weight_max": args.weight_max,
                    "weight_type": "gaussian",
                    "no_mask_probability": args.no_mask_probability,
                    "multi_mask_factor": args.multi_mask_factor,
                    }

    synthetic_training_ad_dataloader = make_synthetic_training_ad_dataloader(dataloader_train,
                                                                                mask_params,
                                                                                amount=0)
    masks = synthetic_training_ad_dataloader.masks

    os.makedirs(os.path.join(os.getcwd(), f"data/{dataset_key}"), exist_ok=True)

    pc_hyperparameters["device"] = ad_device

    algo_class = WrapperPatchCorePlus(**pc_hyperparameters)

    algo_class.enter_dataloaders(dataloader_train, 
                                    dataloader_regular_test, 
                                    dataloader_novel_test)

    algo_class.pre_eval_processing()

    preds = algo_class.eval_outputs_dataloader(synthetic_training_ad_dataloader.to_pass, 
                                                len(synthetic_training_ad_dataloader.to_pass))
    
    # with open(os.path.join(os.getcwd(), f"data/{dataset_key}/train_{dataset_key}.pk"), "wb") as f:
    #     preds = preds[0]['segmentations']
    #     pickle.dump((preds, masks), f)
    
    training_images = preds[0]['segmentations']
    training_masks = torch.vstack(masks)

    training_mean_ = training_images.mean()
    training_std_ = training_images.std()
    
    

    heatmap_set, image_set, targets_set, paths_set = algo_class.test()

    metric_list = ["Imagewise_AUC", "Pixelwise_AUC", "Pixelwise_AUPRO", "PL"]
    scores = run_metrics(metric_list, heatmap_set, image_set, targets_set, paths_set, print_=True) 

    regular_preds = heatmap_set["heatmap_set_regular"]["segmentations"]
    anomalous_preds = heatmap_set["heatmap_set_novel"]["segmentations"]

    regular_image_scores = image_set["image_score_set_regular"]
    anomalous_image_scores = image_set["image_score_set_novel"]

    r_testing_mean_ = regular_preds.mean()
    r_testing_std_ = regular_preds.std()

    a_testing_mean_ = anomalous_preds.mean()
    a_testing_std_ = anomalous_preds.std()

    print("Training mean and std:", training_mean_.item(), training_std_.item())
    print("A testing mean and std:", a_testing_mean_.item(), a_testing_std_.item())
    print("R testing mean and std:", r_testing_mean_.item(), r_testing_std_.item())

    test_targets = targets_set['targets_novel'][:,0]


    training_images = z_norm(training_images, 
                            training_mean_, 
                            training_std_, 
                            center=0.5, 
                            width=0.15)
    anomalous_preds = z_norm(anomalous_preds, 
                            training_mean_, 
                            training_std_, 
                            center=0.5, 
                            width=0.15)
    regular_preds = z_norm(regular_preds, 
                            training_mean_,
                            training_std_, 
                            center=0.5, 
                            width=0.15)
    
    print("Training mean and std:", training_images.mean().item(), training_images.std().item())
    print("A testing mean and std:", anomalous_preds.mean().item(), anomalous_preds.std().item())
    print("R testing mean and std:", regular_preds.mean().item(), regular_preds.std().item())

    paths_anomalous = paths_set["paths_novel"]
    paths_regular   = paths_set["paths_regular"]

    for rotation in [False, True]:
        if rotation:
            unique_key_r = unique_key+"_rot"
        else:
            unique_key_r = unique_key
        cfg_path, dataset_dir = create_dataset(dataset_key=dataset_key, unique_key=unique_key_r)

        train_path_images = os.path.join(dataset_dir, dataset_key+"_"+unique_key_r, "images", "train")
        train_path_labels = os.path.join(dataset_dir, dataset_key+"_"+unique_key_r, "labels", "train")
        os.makedirs(train_path_images, exist_ok=True)
        os.makedirs(train_path_labels, exist_ok=True)

        for image_key, (torch_image, mask) in enumerate(zip(training_images[:,0], 
                                                            training_masks)):
            torch_image[torch_image>1] = 1
            torch_image[torch_image<0] = 0
            image = (torch_image*255).numpy().astype(np.uint8)

            # Step 2: Create a PIL image from the NumPy array
            image = Image.fromarray(image)

            # Step 3: Save the image as a JPEG file
            image.save(os.path.join(train_path_images, f"{image_key}.jpg"))
            
            # bb_coords = extract_bb_coords_mask(mask)
            mask = downscale_mask(mask, 4)
            bb_coords = target_to_rotated_scaled_merged_BBs(np.array(mask), 
                                                            overlap_limit=0.4, 
                                                            min_target_scale=8, 
                                                            rotation=rotation,
                                                            shape=(256, 256))
            if rotation and len(bb_coords)>0:
                bb_coords[:,-1] = np.radians(bb_coords[:,-1])
                
                bb_coords = xywhr2xyxyxyxy(bb_coords)
                bb_coords = torch.stack([item.flatten() for item in bb_coords]).cpu().numpy()
                bb_coords[bb_coords>256] = 256
                bb_coords[bb_coords<0] = 0
            bb_coords = bb_coords/256
            label_path = os.path.join(train_path_labels, f"{image_key}.txt")
            with open(label_path, "w") as f:
                for bb in bb_coords:
                    line = "0 " + " ".join([f"{item:.6g}" for item in bb]) + "\n"
                    f.write(line)

        # preds, masks = None, None
        test_path_images = os.path.join(dataset_dir, dataset_key+"_"+unique_key_r, "images", "test")
        test_path_labels = os.path.join(dataset_dir, dataset_key+"_"+unique_key_r, "labels", "test")
        os.makedirs(test_path_images, exist_ok=True)
        os.makedirs(test_path_labels, exist_ok=True)  
        org_image_path_to_int_label = {}  
        for reg_anom, images, target_group, path_dir, path_set in zip(["regular", "anom"],
                                                [regular_preds[:,0], anomalous_preds[:,0]],
                                                [[None]*len(regular_preds), test_targets],
                                                [dataloader_regular_test.dataset.folder_path, dataloader_novel_test.dataset.folder_path,],
                                                [paths_regular, paths_anomalous]):
            for image_key, (torch_image, mask, file_path) in enumerate(zip(images, 
                                                                            target_group,
                                                                            path_set)):
                original_image_path = os.path.join(path_dir, file_path)

                torch_image[torch_image>1] = 1
                torch_image[torch_image<0] = 0
                image = (torch_image*255).numpy().astype(np.uint8)

                # Step 2: Create a PIL image from the NumPy array
                image = Image.fromarray(image)

                yolo_file_name = f"{reg_anom}_{image_key}.jpg"
                image_path = os.path.join(test_path_images, yolo_file_name)

                
                # Step 3: Save the image as a JPEG file
                image.save(image_path)
                
                if reg_anom=="anom":
                    # bb_coords = extract_bb_coords_mask(mask)
                    mask = downscale_mask(mask, 4)
                    bb_coords = target_to_rotated_scaled_merged_BBs(np.array(mask), 
                                                                    overlap_limit=0.4, 
                                                                    min_target_scale=8, 
                                                                    rotation=rotation,
                                                                    shape=(256, 256))
                    if rotation and len(bb_coords)>0:
                        bb_coords[:,-1] = np.radians(bb_coords[:,-1])
                        
                        bb_coords = xywhr2xyxyxyxy(bb_coords)
                        bb_coords = torch.stack([item.flatten() for item in bb_coords]).cpu().numpy()
                        bb_coords[bb_coords>256] = 256
                        bb_coords[bb_coords<0] = 0
                    
                    bb_coords = bb_coords/256
                    label_path = os.path.join(test_path_labels, f"{reg_anom}_{image_key}.txt")
                    with open(label_path, "w") as f:
                        for bb in bb_coords:
                            line = "0 " + " ".join([f"{item:.6g}" for item in bb]) + "\n"
                            f.write(line)
                else:
                    label_path = os.path.join(test_path_labels, f"{reg_anom}_{image_key}.txt")
                    with open(label_path, "w") as f:
                        pass

                org_image_path_to_int_label[yolo_file_name] = (original_image_path, image_path, label_path)

        org_image_path_to_int_label_path = os.path.join(os.getcwd(), 
                                                        "ultralytics", 
                                                        "cfg", 
                                                        "datasets", 
                                                        dataset_key+"_"+unique_key_r+"_im2im_paths.json")
        with open(org_image_path_to_int_label_path, "w") as f:
            json.dump(org_image_path_to_int_label, f)

        if rotation:
            print("Written OBB yolo dataset to:")
            print(os.path.join(dataset_dir, dataset_key+"_"+unique_key_r))
        else:
            print("Written yolo dataset to:")
            print(os.path.join(dataset_dir, dataset_key+"_"+unique_key_r))

        if args.results_key:
            print("Getting sent:")
            print(dataset_key+"_"+unique_key_r, args.results_key, ad_device, "args", rotation)
            train_yolo_main(dataset_key+"_"+unique_key_r, args.results_key, ad_device, args, rotation=rotation)


if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Compute AP50 and mAP (AP50:95) for YOLOv8 model predictions.")

    # Add arguments
    parser.add_argument('--dataset_key', type=str, required=True, help='Dataset key.')
    parser.add_argument('--ad_device', type=str, required=True, help='GPU identifier - etc "cuda:0"')
    parser.add_argument('--unique_key', type=str, required=True, help='Attached to dataset name to create unique idenfitier for data created')
    parser.add_argument('--weight_min', type=float, default=0.1, help='weight_min')
    parser.add_argument('--weight_max', type=float, default=0.5, help='weight_max')
    parser.add_argument('--results_key', type=str, default="", help='Dir for YOLO results, if provided YOLO will run')
    parser.add_argument('--model', type=str, default='yolov8n', required=False, help='model key')
    parser.add_argument('--no_mask_probability', type=float, default=0.1, required=False, help='prob that synthetic image has no mask')
    parser.add_argument('--multi_mask_factor', type=float, default=0.6, required=False, help='recursive probability of adding another anomaly to a sythnetic image')
    
    # Parse arguments
    args = parser.parse_args()

    main(args.dataset_key, args.ad_device, args.unique_key, args)