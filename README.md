### Data - download the MVTec and/or VisA datasets, add the paths to the following variables in the visionad_wrapper/data/configure_dataset.py files

### Quickstart

python3 create_all_files_run_yolo.py --ad_device "cuda:0" --unique_key "initial_run" --results_key "initial_results"

or

python3 create_files_run_yolo.py --dataset_key mvtec_bottle --ad_device "cuda:0" --unique_key "initial_run" --results_key "initial_results"

## View results here

View results by entering the relevant dataset, unique_key, and results_key into the functions in notebooks/viewing_metrics.ipynb and notebooks/viewing_results.ipynb

### Code description

The code comes in two parts.

1. One part of the code creates the training and testing files for YOLO, i.e. it creates the synthetic anomalies, trains a Patchcore anomaly detection model, and creates the Patchcore outputs. These outputs can be used to train and test the YOLO model.

2. The YOLO model is trained and tested using the files in the aforementioned process.

This seperation of concerns allows us to create one set of data using a specific synthetic anomaly parameters, and experiment with many different sets YOLO training parameters. However, the code allows you to run both steps at once if you desire.

# create_files_run_yolo.py

- **`--dataset_key`** (type: `str`, required: `True`)  
  Dataset name

- **`--ad_device`** (type: `str`, required: `True`)  
  GPU identifier - etc "cuda:0"

- **`--unique_key`** (type: `str`, required: `True`)  
  A unique key which will be attached to the dataset name, which is used to identify the data created.

- **`--weight_min`** (type: `float`, default: `0.1`)  
  Specifies the minimum weight to apply during calculations. Default is 0.1.

- **`--weight_max`** (type: `float`, default: `0.5`)  
  Specifies the maximum weight to apply during calculations. Default is 0.5.

- **`--results_key`** (type: `str`, required: `False`)  
  If provided, the YOLO will automatically run on the data created. The directly name for the YOLO results.

- **`--model`** (type: `str`, default: `'yolov8n'`, required: `False`)  
  Defines the YOLO model key to be used. Default is 'yolov8n'. Only relevant is results key is provided.

This script generates training and testing files using synthetic anomalies and the Patchcore algorithm. It creates files for both regular and oriented bounding boxes automatically.

This automatically creates data with normal BBs and OBBs. The normal data has the key {unique_key}, whilst the orientated data has the key {unique_key}\_rot .

If a `results_key` is provided, the script will also train and test the YOLO detection.

# create_all_files_run_yolo.py

Similar to the above script, but loops through all the dataset classes in MVTec and VisA, creating the training and testing files. As above, if results_key is provided, the script will train and test the YOLO detection.

These scripts create saves the data to a folder named: {dataset_key}\_{unique_key}. This key is used by the YOLO scripts to find the relevent data.

# train_yolo_model.py

- **`--dataset_key`** (type: `str`, required: `True`)  
  Dataset name

- **`--ad_device`** (type: `str`, required: `True`)  
  GPU identifier - etc "cuda:0"

- **`--unique_key`** (type: `str`, required: `True`)  
  The unique key used to make the data in create_files_run_yolo.py

- **`--results_key`** (type: `str`, required: `True`)  
  The directly name for these results

- **`--weight_min`** (type: `float`, default: `0.1`)  
  The minimum weight applied during calculations. Default is `0.1`.

- **`--weight_max`** (type: `float`, default: `0.5`)  
  The maximum weight applied during calculations. Default is `0.5`.

- **`--model`** (type: `str`, default: `'yolov8n'`)  
  The YOLO model key to use for the operation. Default is `'yolov8n'`.

This file runs the trains and tests the YOLO network on the files corresponding to the specific dataset and unique_key.

Add "\_rot" to the end of the unique_key to run the OBB YOLO.

# train_all_yolo_models.py

This file runs the trains and tests the YOLO network on the files of all datasets that correspond to the unique_key.
