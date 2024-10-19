import sys
from train_yolo_model import main as train_yolo_main

import argparse
parser = argparse.ArgumentParser(description="Compute AP50 and mAP (AP50:95) for YOLOv8 model predictions.")

# Add arguments
parser.add_argument('--ad_device', type=str, required=True, help='GPU identifier - etc "cuda:0"')
parser.add_argument('--unique_key', type=str, required=True, help='unique_key')
parser.add_argument('--results_key', type=str, default="standard", required=False, help='directly name that these results go into')
parser.add_argument('--model', type=str, default='yolov8n', required=False, help='model key')

args = parser.parse_args()

dataset_list = ['mvtec_bottle',
                 'mvtec_cable',
                 'mvtec_capsule',
                 'mvtec_carpet',
                 'mvtec_grid',
                 'mvtec_hazelnut',
                 'mvtec_leather',
                 'mvtec_metal_nut',
                 'mvtec_pill',
                 'mvtec_screw',
                 'mvtec_tile',
                 'mvtec_toothbrush',
                 'mvtec_transistor',
                 'mvtec_wood',
                 'mvtec_zipper',
                 'visa_pipe_fryum',
                 'visa_pcb4',
                 'visa_pcb3',
                 'visa_pcb2',
                 'visa_pcb1',
                 'visa_macaroni2',
                 'visa_macaroni1',
                 'visa_fryum',
                 'visa_chewinggum',
                 'visa_cashew',
                 'visa_capsules',
                 'visa_candle',
]

for dataset_key in dataset_list:
    # main(dataset_key, ad_device, unique_key)
    if args.unique_key[-4:]=="_rot":
        rotation = True
    else:
        rotation = False
    try:
        train_yolo_main(dataset_key+"_"+args.unique_key, args.results_key, args.ad_device, args, rotation)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        