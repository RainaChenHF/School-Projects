#!/usr/bin/env python
# coding: utf-8

# # 1. Setting Up the Environment

# In[ ]:


get_ipython().system('pip install cvzone')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install git+https://github.com/ultralytics/ultralytics.git')
get_ipython().system('pip install optuna')
get_ipython().system('pip install albumentations')
get_ipython().system('pip install pyyaml')


# In[ ]:


import json
import cvzone
import cv2
import os
import uuid
import random
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from google.colab import drive
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
import os
import json

# Set seed for reproducibility
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)


# # 2. Load and Prepare the Dataset
# This part is crucial for you to change accordingly

# In[ ]:


#RUN IF RUNNING IN GOOGLE COLAB
drive.mount('/content/drive')
get_ipython().system('cp /content/drive/MyDrive/DL2025/YoloV8/scarecrow_dataset.zip /content/ # change to appropriate directory where dataset is')
get_ipython().system('unzip -q /content/drive/MyDrive/DL2025/YoloV8/scarecrow_dataset.zip -d /content/custom_data # chnage first part to appropiate directory, unzip it and call it custom_data')


# In[ ]:


# All of these should be changed accordingly to where these files are found
# if notebook is run on google colab the following paths should work for the rest of notebook
base_dir = "/home/filip/code/dani/scarecrow_dataset" # should be directory of where the data is saved (unzipped version)
path_to_train_set = "/home/filip/code/dani/scarecrow_dataset/train" # path to train set folder
path_to_val_set = "/home/filip/code/dani/scarecrow_dataset/val" # path to validation set folder
path_to_test_set = "/home/filip/code/dani/scarecrow_dataset/test" # path to test set folder
test_set_path_images = '/home/filip/code/dani/scarecrow_dataset/test/images' # path to images folder of the test set folder

model_path = "/home/filip/code/dani/runs/detect/train2/weights/best.pt" # path to best.pt model


# # 3. Convert Annotations from JSON to YOLO Format

# In[ ]:


def process_dataset(split, base_dir=base_dir):
    """Reading anotations from json file and converting them into
    YOLO format. Within the corresponding data split a labels folder is
    created where each image has its corresponding txt file.
    """
    base_dir = f"{base_dir}/{split}" # define what data split you want to perform this on
    img_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels") # making a new labels directory
    os.makedirs(label_dir, exist_ok=True)

    # getting information from json file
    with open(os.path.join(base_dir, "annotations.json"), 'r') as f:
        items = json.load(f)
    # Adding a bar that tells you percentage of labels being done for data split
    for item in tqdm(items, desc=f"Processing {split}"):
        file_name = item['OriginalFileName']
        img_path = os.path.join(img_dir, file_name)

        # Load image to get dimensions
        image = cv2.imread(img_path)
        if image is None: # added in case of an error
            print(f"Warning: couldn't load {img_path}")
            continue

        # extracting height and width of each image
        h, w = image.shape[:2]

        # Extracting bounding boxes and convert to YOLO format
        yolo_lines = []
        for ann in item.get("AnnotationData", []):
            coords = ann["Coordinates"]
            x_min = int(min(pt["X"] for pt in coords))
            y_min = int(min(pt["Y"] for pt in coords))
            x_max = int(max(pt["X"] for pt in coords))
            y_max = int(max(pt["Y"] for pt in coords))

            # Converting to YOLO format (normalized coordinates)
            x_center = (x_min + x_max) / (2 * w)
            y_center = (y_min + y_max) / (2 * h)
            bbox_width = (x_max - x_min) / w
            bbox_height = (y_max - y_min) / h

            # Ensure values are valid (between 0 and 1), necessary for yolo format
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            bbox_width = max(0, min(1, bbox_width))
            bbox_height = max(0, min(1, bbox_height))

            # each row in the text file corresponds to a bird in the image
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Save YOLO format labels
        base_name = os.path.splitext(file_name)[0]
        with open(os.path.join(label_dir, base_name + ".txt"), 'w') as f:
            f.write("\n".join(yolo_lines))

# Process train, validation, test sets
process_dataset("train")
process_dataset("val")
process_dataset("test")


# # 4. Trainning YOLOv8 model

# In[ ]:


### 1. Creating a data.yaml for YOLOv8 Trainning Configuration ################
def create_data_yaml(output_path="data.yaml"):
  """ Creating a data.yaml file for YOLOv8 training configuration.
  So model knows where to get the training, validation, and test data.
  """
  data = {
        'train': path_to_train_set, # path to train set
        'val': path_to_val_set,     # path to val set
        'test':path_to_test_set,    # path to test set
        'nc': 1, # number of classes, 1 as we are merely identifying birds
        'names': ['bird'] # name of class
    }

  with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        print(f"Created {output_path}")


create_data_yaml()

### 2. Training Configuration YAML for YOLOv8 and Training######################
# Please refer to the appendix to see how optuna was utilized to define best set of hyperparameters.


def create_combined_yaml(output_path="train_config.yaml"):
    """
    Creates a training configuration YAML file using the best hyperparameters
    discovered by Optuna. This config is intended for a full training run with:
    - More epochs to allow deeper learning
    - Optimal augmentation and training settings for small object detection (e.g., birds)

    The resulting YAML file combines both model and training parameters in one place.
    """

    config = {
        'deterministic': True,
        'seed':42,
        'task': 'detect',
        'mode': 'train',
        'model': 'yolov8m.pt',               # changed to a larger model for better results
        'data': 'data.yaml',        # Path to dataset configuration
        'epochs': 120,                       # Increased for extended training
        'imgsz': 1024,                       # High-resolution input to improve small object detection
        'patience': 20,                      # Early stopping after 20 epochs of no improvement
        'batch': 8,                          # set to 8 bc that is as much as our GPU can do


        # Optimized hyperparameters from Optuna
        'lr0': 0.0002092307436660672,        # Initial learning rate
        'lrf': 0.8678870531410413,           # Final learning rate fraction
        'optimizer': 'Adam',                 # Best optimizer selected
        'weight_decay': 0.001603940700102649,

        # Data augmentation settings (following a minimal approacg)
        'hsv_h': 0.015,                      # Slight hue shift (good for varying lighting)
        'hsv_s': 0.7,                        # Strong saturation variation
        'hsv_v': 0.4,                        # Moderate brightness variation
        'fliplr': 0.5,                       # Lateral flip makes sense for aerial bird views

        # Augmentations disabled based on Optuna recommendations or dataset characteristics
        'degrees': 0.0,                      # Rotation not helpful
        'translate': 0.0,                    # Can crop out small objects
        'scale': 0.0,                        # No scaling
        'flipud': 0.0,                       # Birds are rarely upside-down

        # Disable complex augmentations not as suitable for small object detection (based on documentation and research)
        'shear': 0.0,                        # Not beneficial
        'perspective': 0.0,                  # Avoid distortion (small objects get deformed)
        'bgr': 0.0,                          # BGR swap disabled
        'mosaic': 0.0,                       # Disabled; can occlude small objects
        'mixup': 0.0,                        # Disabled; not good for small targets
        'cutmix': 0.0,                       # Also not suitable for small object detection
        'copy_paste': 0.0,                   # Requires segmentation masks
        'erasing': 0.0                       # Risk of erasing entire small objects like birds
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Created training config: {output_path}")

create_combined_yaml()


### 3. Trainnning model #######################################################
get_ipython().system('yolo cfg=train_config.yaml seed=42')


# # 5. YOLOv8 Validation mode
# To choose appropriate confidence threshold

# In[ ]:


# Load model
model = YOLO(model_path)


results_val = model.val(
    data='data.yaml',
    split='val',  # Use validation set for tuning
    iou=0.55,
    verbose=False
)

# confidence was visually chosen to be 0.2 (appropiate mix of recall and precision we were going for)


# In[ ]:


# based on selected threshold running it on test set
results_test = model.val(
        data='data.yaml',
        split='test',  # Use validation set for tuning
        conf=0.2,
        iou=0.55,
        verbose=False
    )


# # 6. Run Inference on Test Set

# In[ ]:


# Run inference on the test set with a confidence threshold of 0.158
model = YOLO(model_path)
results = model.predict(
    source=test_set_path_images,
    conf=0.2,  # Use best confidence from validation
    iou=0.55,
    save=True  # This will save annotated images in /runs/detect/predict/
)


# In[ ]:


import glob
# to visulaize all the images in the test set with their predictions
# Display the first 10 images with predictions
from IPython.display import Image, display
for image_path in glob.glob(f'/home/filip/code/dani/runs/detect/predict/*.jpg')[:10]:
    display(Image(filename=image_path, height=400))
    print('\n')


# # 7. JSON file with predictions on Test set

# In[ ]:


# Save each image's prediction result as a separate JSON file
json_output_dir = "yolo_json_output"
os.makedirs(json_output_dir, exist_ok=True)

for i, result in enumerate(results):
    json_str = result.to_json()
    with open(os.path.join(json_output_dir, f"result_{i}.json"), "w") as f:
        f.write(json_str)

# Merge all individual JSON files into one combined JSON list
merged_results = []

for filename in sorted(os.listdir(json_output_dir)):
    if filename.endswith(".json") and filename.startswith("result_"):
        filepath = os.path.join(json_output_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_results.extend(data)
            else:
                merged_results.append(data)

# Save the merged result into a single JSON file
with open("merged_predictions.json", "w") as f:
    json.dump(merged_results, f, indent=2)

print("Inference complete! Merged results saved to merged_predictions.json")


# # 8. Benchmarking the model
# 

# In[ ]:


from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model=model_path, data="data.yaml", imgsz=1024, half=False, device=0)


# ## Appendix: Optimize YOLOv8 Training Parameters Using Optuna Bayesian Search
# 
# Code below was used to determine best hyperparameters of final model
# 
# To check best hyperparameter combination found scroll to last bit of output of the following code chunk
# 

# In[ ]:


import optuna
from ultralytics import YOLO
import os

# Fixed hyperparameters
FIXED_PARAMS = {
    "patience": 10, # early stop, if the model does not improve adter 10 epochs stop trainning
    "epochs": 50,   # total number of epochs per trial
    "imgsz": 1024,  # need to be multiples of 32 so set to a high resolution for better results
    "batch": -1,                # 60% of GPU utilization
    'shear': 0.0,               # disabled, as other techniques more appropiate
    'perspective': 0.0,         # disabled, makes small objects become very unclear. Image distortion does not seem to be an issue based on images in dataset
    'flipud': 0.0,              # disabled, not the most applicable when detecting birds
    'bgr': 0.0,                 # disabled, there are better techniques
    'close_mosaic': 20,         # mosaic makes training more complex so would not run for 20 last epochs
    'mixup': 0.0,               # disabled as overlap may hide small objects completely
    'cutmix': 0.0,              # disabled, other techniques are more appropriate for small objects
    'copy_paste': 0.0,          # disabled as need segmentation masks (segmentation-specifc task)
    'erasing': 0.0              # disabled, might delete birds from image completely (Classification-Specific Augmentations)

}

# Path to dataset
DATA_PATH = "/content/data.yaml"

def objective(trial):
    # Core parameters with most impact, trying different combinations of these each trial
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.1, 1.0)
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01)
    # Tune momentum only for SGD
    if optimizer == "SGD":
        momentum = trial.suggest_float("momentum", 0.8, 0.98)
    else: # not applicable to Adam
        momentum = None

    # experimenting with different data augmentation strategies
    # minimal makes minimal changes (only does horizontal flip, moderate enables other techniques,
    # while aggressive has higher values (more aggressive changes))
    aug_strategy = trial.suggest_categorical("aug_strategy", ["minimal", "moderate", "aggressive"])

    if aug_strategy == "minimal":
        mosaic, fliplr, degrees, scale = 0.0, 0.5, 0.0, 0.0
    elif aug_strategy == "moderate":
        mosaic, fliplr, degrees, scale = 0.3, 0.5, 10.0, 0.1
    else:  # aggressive
        mosaic, fliplr, degrees, scale = 0.5, 0.5, 15.0, 0.2

    # Other parameters set to reasonable defaults
    # these are always included in each trial
    hsv_h, hsv_s, hsv_v = 0.015, 0.7, 0.4

    # Load small model
    model = YOLO("yolov8s.yaml")

    # Train with suggested + fixed parameters
    results = model.train(
        data=DATA_PATH,
        seed=42,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        scale=scale,
        fliplr=fliplr,
        mosaic=mosaic,
        patience=FIXED_PARAMS["patience"],
        epochs=FIXED_PARAMS["epochs"],
        imgsz=FIXED_PARAMS["imgsz"],
        batch=FIXED_PARAMS["batch"],
        shear=FIXED_PARAMS["shear"],
        perspective=FIXED_PARAMS["perspective"],
        flipud=FIXED_PARAMS["flipud"],
        bgr=FIXED_PARAMS["bgr"],
        close_mosaic=FIXED_PARAMS["close_mosaic"],
        mixup=FIXED_PARAMS["mixup"],
        cutmix=FIXED_PARAMS["cutmix"],
        copy_paste=FIXED_PARAMS["copy_paste"],
        erasing=FIXED_PARAMS["erasing"],
        verbose=False
    )

    # MAP is the metric to maximize
    return results.box.map


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, timeout=7200)

# Best trial results
print("Best hyperparameters:", study.best_trial.params)



# In[ ]:


print("Best trial number:", study.best_trial.number)
print("Best value (objective):", study.best_trial.value)

