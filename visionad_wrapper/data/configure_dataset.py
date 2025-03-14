import os
from re import S
import torch
from visionad_wrapper.data.utils import add_dataset, configure_mvtec_class

if os.name == "nt":
    MVTEC_PATH = r"D:\Documents\PhD_work\supporting_data\MVTec_Test\mvtec_anomaly_detection.tar\mvtec_anomaly_detection"
    IDW_PATH   = r"D:\Documents\PhD_work\datasets\youtube_datasets\IDW_camera_ready_final_6"
    VISA_PATH  = r":("
    STREETHAZARDS_PATH = r":("
    RESULTS_DIRECTORY = r"D:\Documents\PhD_work\random\TEST_RESULTS"
else:
    MVTEC_PATH = r"/mnt/faster0/adjt20/benchmark_datasets/mvtec_anomaly_detection"
    IDW_PATH   = r"/mnt/faster0/adjt20/benchmark_datasets/IDW_camera_ready_final_6"
    VISA_PATH  = r"/mnt/faster0/adjt20/benchmark_datasets/visa/spot-diff/VisA_pytorch/1cls"
    STREETHAZARDS_PATH = r"/mnt/faster0/adjt20/benchmark_datasets/streethazards"
    RESULTS_DIRECTORY = r"/mnt/lime/homes/adjt20/results"



if MVTEC_PATH=="":
    raise NotImplementedError("Add MVTEC_PATH in data.configure_dataset.py")
if RESULTS_DIRECTORY=="":
    raise NotImplementedError("Add RESULTS_DIRECTORY in data.configure_dataset.py")

datasets = {}

datasets["demo_dataset"] = {"folder_path_regular_train" : r"",
                            "folder_path_regular_test": r"",
                            "folder_path_novel_test"  : r"",
                            "folder_path_novel_masks"  : r"",
                            "regular_train_start" : None,
                            "regular_train_end"   : None,
                            "regular_test_start"  : None, 
                            "regular_test_end" : None,
                            "novel_test_start" : None, 
                            "novel_test_end" : None,
                            "skip_train_regular" : 1, 
                            "skip_test_regular"  : 1, 
                            "skip_test_novel"    : 1, 
                            "mean"         : [0.485, 0.456, 0.406],
                            "std"          : [0.229, 0.224, 0.225],
                            "pixel_labels" :  True,
}

mvtec_folders = ['bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',    
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper'
]

idw_folders = ["BirdStrike",
               "CarBody",
               "HouseStructure",
               "HeadGasket",
               "NozzleGuideVane",
               "InspectionPipes",
               "Combined",]

visa_folders = ["pipe_fryum",
                "pcb4",
                "pcb3",
                "pcb2",
                "pcb1",
                "macaroni2",
                "macaroni1",
                "fryum",
                "chewinggum",
                "cashew",
                "capsules",
                "candle",
        ]

for name in mvtec_folders:
    if not os.path.exists(os.path.join(MVTEC_PATH, name, "anomaly_test_grouped")) or not os.path.exists(os.path.join(MVTEC_PATH, name, "ground_truth_grouped")):
        ## the MVTec class it its raw format is not suitable for the wrapper, as the defects and GTs are stored in defect-class specific folders
        ## this function copies all the defects and GTs into two respective folders, which is suitable for the wrapper
        try:
            configure_mvtec_class(os.path.join(MVTEC_PATH, name))
        except:
            pass

    datasets["mvtec_"+name] = {"dataset_key": "mvtec_"+name,
                               "folder_path_regular_train" : os.path.join(MVTEC_PATH, name, "train", "good"),
                               "folder_path_regular_test" : os.path.join(MVTEC_PATH, name, "test", "good"),
                               "folder_path_novel_test" : os.path.join(MVTEC_PATH, name, "anomaly_test_grouped"),
                               "folder_path_novel_masks" : os.path.join(MVTEC_PATH, name, "ground_truth_grouped"),
                               "regular_train_start" : None,
                               "regular_train_end" : None,
                               "regular_test_start" : None, 
                               "regular_test_end" : None,
                               "novel_test_start" : None, 
                               "novel_test_end" : None, 
                               "mean"         : [0.485, 0.456, 0.406],
                               "std"          : [0.229, 0.224, 0.225],
                               }

for name in idw_folders:
    datasets[name] = {"dataset_key": name,
                        "folder_path_regular_train" : os.path.join(IDW_PATH, name, "train", "good"),
                        "folder_path_regular_test" : os.path.join(IDW_PATH, name, "test", "good"),
                        "folder_path_novel_test" : os.path.join(IDW_PATH, name, "test", "anomalous"),
                        "folder_path_novel_masks" : os.path.join(IDW_PATH, name, "ground_truth", "anomalous"),
                        "regular_train_start" : None,
                        "regular_train_end" : None,
                        "regular_test_start" : None, 
                        "regular_test_end" : None,
                        "novel_test_start" : None, 
                        "novel_test_end" : None, 
                        "mean"         : [0.485, 0.456, 0.406],
                        "std"          : [0.229, 0.224, 0.225],
                        }


for name in visa_folders:
    datasets["visa_"+name] = {"dataset_key": name,
                        "folder_path_regular_train" : os.path.join(VISA_PATH, name, "train", "good"),
                        "folder_path_regular_test" : os.path.join(VISA_PATH, name, "test", "good"),
                        "folder_path_novel_test" : os.path.join(VISA_PATH, name, "test", "bad"),
                        "folder_path_novel_masks" : os.path.join(VISA_PATH, name, "ground_truth", "bad"),
                        "regular_train_start" : None,
                        "regular_train_end" : None,
                        "regular_test_start" : None, 
                        "regular_test_end" : None,
                        "novel_test_start" : None, 
                        "novel_test_end" : None, 
                        "mean"         : [0.485, 0.456, 0.406],
                        "std"          : [0.229, 0.224, 0.225],
                        }


datasets["streethazards"] = {"folder_path_regular_train" : os.path.join(STREETHAZARDS_PATH, r"train", "images", "training_grouped"),
                            "folder_path_regular_test": os.path.join(STREETHAZARDS_PATH, "test", "regular_test"),
                            "folder_path_novel_test"  : os.path.join(STREETHAZARDS_PATH, r"test/anomaly_test_grouped"),
                            "folder_path_novel_masks"  : os.path.join(STREETHAZARDS_PATH, r"test/ground_truth_grouped"),
                            "regular_train_start" : None,
                            "regular_train_end"   : None,
                            "regular_test_start"  : None,
                            "regular_test_end" : None,
                            "novel_test_start" : None,
                            "novel_test_end" : None,
                            "skip_train_regular" : 1,
                            "skip_test_regular"  : 1,
                            "skip_test_novel"    : 1,
                            "mean"         : [0.485, 0.456, 0.406],
                            "std"          : [0.229, 0.224, 0.225],
                            "pixel_labels" :  True,
}

def load_dataset(key):
    return datasets[key]




    




