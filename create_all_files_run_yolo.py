from create_files_run_yolo import main
import sys

import argparse
parser = argparse.ArgumentParser(description="Compute AP50 and mAP (AP50:95) for YOLOv8 model predictions.")

# Add arguments
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
    main(dataset_key, args.ad_device, args.unique_key, args)
    # train_yolo_main(dataset_key+"_"+unique_key, ad_device)
    # try:
    #     train_yolo_main(dataset_key+"_"+unique_key, ad_device)
    # except Exception as e:
    #     import traceback
    #     print(traceback.format_exc())
        