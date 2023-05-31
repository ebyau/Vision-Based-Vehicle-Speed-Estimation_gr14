# Miscellaneous
import matplotlib.pyplot as plt
import numpy as np
import time, os, sys, torch
from os.path import zfill

# torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision.utils import save_image
from torchvision import datasets, transforms

# Custom
from model import dlavNet
from dataset import KittiDataset, DlavDataset

USE_GPU = False

# PIL transform
to_pil = transforms.ToPILImage()

# Select device
device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")
print("Using {} device ..".format(device))

# Accuracy are available here : https://github.com/isl-org/MiDaS#Accuracy
depth_model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#depth_model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#depth_model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

depth_net = torch.hub.load("intel-isl/MiDaS", depth_model_type)

# Transform to get the correct input format
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if depth_model_type == "DPT_Large" or depth_model_type == "DPT_Hybrid":
  depth_transform = midas_transforms.dpt_transform
else:
  depth_transform = midas_transforms.small_transform

depth_net.to(device)
depth_net = depth_net.eval() # Set into evaluation mode

# Dataset directory
dataset_dir = '/dataset_kitti'

# Set seed
torch.manual_seed(0)

kitti_dataset = KittiDataset(dataset_dir, load_for_example=True)
for (_, img_pair, _, drive, idx_sample) in kitti_dataset:
  if not os.path.exists(dataset_dir + '/' + drive + '/depth'):
    depth_folder = dataset_dir + '/' + drive + '/depth'
    os.mkdir(depth_folder)
    depth_folder = depth_folder + '/data'
    os.mkdir(depth_folder)
  else:
    depth_folder = dataset_dir + '/' + drive + '/depth/data'

  # Convret to PIL to feed into depth net
  img_pair_pil = [to_pil(img) for img in img_pair]
  single_img_pil = img_pair_pil[1]

  #################### APPLY DEPTH TO FIRST IMAGE OF EACH PAIR ##################
  depth_transformed_img = depth_transform(np.array(single_img_pil))
  depth_single_img = depth_net(depth_transformed_img.to(device))
  depth_single_img = torch.nn.functional.interpolate(
      depth_single_img.unsqueeze(1),
      size=img_pair[0].shape[1:3],
      mode="bicubic",
      align_corners=False,
  ).squeeze()

  save_image(depth_single_img / depth_single_img.max(), depth_folder + '/' + str(idx_sample+1).zfill(10) + '.png')
  torch.save(depth_single_img, depth_folder + '/' + str(idx_sample+1).zfill(10) + '.pt')

  print(drive + "_" + str(idx_sample+1))













