# Miscellaneous
import matplotlib.pyplot as plt
import numpy as np
import time, sys, torch, os
from os.path import zfill

# torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision.utils import save_image, flow_to_image
from torchvision import datasets, transforms
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F

# Custom
from model import dlavNet
from dataset import KittiDataset, DlavDataset

USE_GPU = False

# Select device
device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")
print("Using {} device ..".format(device))

# Get RAFT model
weights = Raft_Large_Weights.DEFAULT
raft_transforms = weights.transforms()

raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)

raft.to(device)
raft = raft.eval()

# Pre-processing for the optical flow
def optical_flow_preprocess(img1_batch, img2_batch):
  img1_batch = F.resize(img1_batch, size=[360, 1224], antialias=False)
  img2_batch = F.resize(img2_batch, size=[360, 1224], antialias=False)
  return raft_transforms(img1_batch, img2_batch)

# Dataset directory
dataset_dir = '/dataset_kitti'

# Set seed
torch.manual_seed(0)

kitti_dataset = KittiDataset(dataset_dir, load_for_example=True)
for (_, img_pair, _, drive, idx_sample) in kitti_dataset:
  if not os.path.exists(dataset_dir + '/' + drive + '/optical_flow'):
    of_folder = dataset_dir + '/' + drive + '/optical_flow'
    os.mkdir(of_folder)
    of_folder = of_folder + '/data'
    os.mkdir(of_folder)
  else:
    of_folder = dataset_dir + '/' + drive + '/optical_flow/data'

  img1, img2 = optical_flow_preprocess(img_pair[0].unsqueeze(0), img_pair[1].unsqueeze(0))
  optical_flow = raft(img1.to(device), img2.to(device))[-1] # Optical flow has several iterations -> keep the last

  optical_flow_img = flow_to_image(optical_flow)

  save_image(optical_flow_img[0]  / 255., of_folder + '/' + str(idx_sample+1).zfill(10) + ".png")
  torch.save(optical_flow.squeeze(), of_folder + '/' + str(idx_sample+1).zfill(10) + ".pt")

  print(drive + "_" + str(idx_sample+1))













