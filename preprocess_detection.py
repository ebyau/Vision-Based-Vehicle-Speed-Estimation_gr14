# Miscellaneous
import matplotlib.pyplot as plt
import numpy as np
import time, sys, torch
from ultralytics import YOLO

# torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F

# Custom
from model import dlavNet
from dataset import KittiDataset, DlavDataset

USE_GPU = False

# PIL transform
to_pil = transforms.ToPILImage()

# Select device
device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")
print("Using {} device ..".format(device))

# Load YOLO model
yolo = YOLO("yolov8x.pt")
yolo.to(device)

# Dataset directory
dataset_dir = '/dataset_kitti'

# Set seed
torch.manual_seed(0)

kitti_dataset = KittiDataset(dataset_dir, load_for_example=True)
for (_, img_pair, _, drive, idx_sample) in kitti_dataset:
  if not os.path.exists(dataset_dir + '/' + drive + '/detection'):
    detection_folder = dataset_dir + '/' + drive + '/detection'
    os.mkdir(detection_folder)
    detection_folder = detection_folder + '/data'
    os.mkdir(detection_folder)
  else:
    detection_folder = dataset_dir + '/' + drive + '/detection/data'

  img_pair_pil = [to_pil(img) for img in img_pair]

  # Predict
  detection = yolo.predict(img_pair_pil[1])
  detection_mask = torch.full((img_pair[0].shape[1], img_pair[0].shape[2]), 1) # tensor full of 1 with same shape as original image

  # Each tensor is : [x1, y1, x2, y2, prob, class]
  # Construct detection mask 
  for cls, box in zip(detection[0].boxes.cls, detection[0].boxes.data):
    rounded_box = box.round()
    rounded_box[4] = box[4] # Round to nearest integer x1, y1, x2, y2 and class
    name = detection[0].names[int(cls)]
    prob = rounded_box[4]
    if name in ['car', 'truck', 'person', 'bicycle', 'bus', 'train'] and prob > 0.7:  # Only remove these classes if prob > 0.7
      detection_mask[rounded_box[0].int():rounded_box[2].int(), rounded_box[1].int():rounded_box[3].int()] = 0 # Set to zero region that contains an object

  torch.save(detection_mask.to(device), detection_folder + '/' + str(idx_sample+1).zfill(10) + ".pt")

  print(drive + "_" + str(idx_sample+1))













