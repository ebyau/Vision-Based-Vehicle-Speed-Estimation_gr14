from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import datetime
from dateutil import parser
import linecache
import json
import numpy as np
import torch

class KittiDataset(Dataset):
  def __init__(self, root_dir, preprocessed_dataset=False, window_size=2, measurement_keywords=['vf', 'vl', 'vu'], device='cpu', load_for_example=False, mode='train', split_ratio=[60,20,20], categories=['all']):
    # parameters
    self.preprocessed = preprocessed_dataset
    self.window_size = window_size
    self.keywords = measurement_keywords
    self.example = load_for_example
    self.zfill_size=10
    self.transform  = transforms.Compose([transforms.CenterCrop((360,1224)), transforms.ToTensor()]) # before : 370, 1224
    self.device = device
    
    # raw directories
    self.root_dir = root_dir
    self.timestamps_file = 'oxts/timestamps.txt'
    self.dataformat_file = 'oxts/dataformat.txt'
    self.oxts_dir = 'oxts/data/'
    self.img_raw_dir = 'image_02/data/'

    # preprocessed directories
    self.img_depth_dir = 'depth/data/'
    self.img_flow_dir = 'optical_flow/data/'
    self.detection_dir = 'detection/data/'

    # get length of each drives   
    if self.preprocessed:
      np.random.seed(6)
      self.drive_directories = np.array([],dtype=str)
      self.drives_size = np.array([],dtype=np.int32)
      for category_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, category_dir)):
          if (category_dir in categories) or categories[0]=='all':
            drive_directories = [os.path.join(category_dir, date_dir, drive_dir) \
                                for date_dir in os.listdir(os.path.join(root_dir, category_dir)) if os.path.isdir(os.path.join(root_dir, category_dir, date_dir)) \
                                for drive_dir in os.listdir(os.path.join(root_dir, category_dir, date_dir)) if os.path.isdir(os.path.join(root_dir, category_dir, date_dir, drive_dir))]
            np.random.shuffle(drive_directories)
            drives_size = [int(len(os.listdir(os.path.join(root_dir, drive_dir, self.img_flow_dir)))/2)-self.window_size + 1 \
                            for drive_dir in drive_directories]     
            cumulative_drives_size = np.cumsum(drives_size)
            tot_frames = cumulative_drives_size[-1]

            size1= int(split_ratio[0]*tot_frames/100)
            size2 = int(split_ratio[1]*tot_frames/100)
            size3 = int(split_ratio[2]*tot_frames/100)

            idx0 = 0
            idx1 = np.searchsorted(cumulative_drives_size, size1, side='right')
            if idx1==idx0:
              idx1+=1
            idx2 = np.searchsorted(cumulative_drives_size, size2+cumulative_drives_size[idx1-1], side='right')
            if idx2==idx1:
              idx2+=1
            if np.sum(split_ratio)==100:
              idx3=len(cumulative_drives_size)
            else:
              idx3 = np.searchsorted(cumulative_drives_size, size3+cumulative_drives_size[idx2-1], side='right')
            if idx3==idx2:
              idx3+=1
            if idx3>len(cumulative_drives_size):
              print('Split ratio cannot be satisfied')
              idx0, idx1, idx2, idx3 = 0,0,0,0
            
            if mode=='train':
              idx_first = idx0
              idx_end = idx1
            elif mode=='val':
              idx_first = idx1
              idx_end = idx2
            elif mode=='test':
              idx_first = idx2
              idx_end = idx3
            else:
              print(f'Mode Error: {mode} not supported. Use [train/val/test]')
            self.drive_directories = np.concatenate((self.drive_directories, drive_directories[idx_first:idx_end]))
            self.drives_size = np.concatenate((self.drives_size, drives_size[idx_first:idx_end]))
    else:
      self.drive_directories = [os.path.join(category_dir, date_dir, drive_dir) \
                              for category_dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, category_dir))\
                              for date_dir in os.listdir(os.path.join(root_dir, category_dir)) if os.path.isdir(os.path.join(root_dir, category_dir, date_dir)) \
                              for drive_dir in os.listdir(os.path.join(root_dir, category_dir, date_dir)) if os.path.isdir(os.path.join(root_dir, category_dir, date_dir, drive_dir))]
      self.drives_size = [len(os.listdir(os.path.join(root_dir, drive_dir, self.img_raw_dir)))-self.window_size + 1 \
                          for drive_dir in self.drive_directories]
    self.cumulative_drives_size = np.cumsum(self.drives_size)

    # get index of measurements
    dataformat_file = os.path.join(root_dir, self.drive_directories[0], self.dataformat_file)
    with open(dataformat_file) as f:
      lines = f.readlines()
    keys = [line.split(':')[0].strip() for line in lines]
    self.idx_keywords = [keys.index(keyword) for keyword in self.keywords]

  def __len__(self):
    return self.cumulative_drives_size[-1]
  
  def __getitem__(self, idx):
    # search which drives has corresponding idx
    idx_drive = np.searchsorted(self.cumulative_drives_size,idx,side='right')
    drive = self.drive_directories[idx_drive]
    idx_sample = idx - self.cumulative_drives_size[idx_drive -1] if idx_drive > 0 else idx


    # update all directories / paths
    drive_dir = os.path.join(self.root_dir, drive)
    timestamp_file = os.path.join(drive_dir, self.timestamps_file)
    oxts_dir =  os.path.join(drive_dir, self.oxts_dir)
    img_raw_dir = os.path.join(drive_dir, self.img_raw_dir)
    img_depth_dir =  os.path.join(drive_dir, self.img_depth_dir)
    img_flow_dir =  os.path.join(drive_dir, self.img_flow_dir)
    detection_dir =  os.path.join(drive_dir, self.detection_dir)

    # get all data in window size
    timestamps = []
    oxts = []
    img_raw = []
    img_depth = []
    img_flow = []
    detection = []
    for i in range(self.window_size):
      if self.preprocessed:
        idx = idx_sample + i
        idx_string =  str(idx+1).zfill(10)

        oxts_path = os.path.join(oxts_dir, idx_string + '.pt')
        oxts.append(torch.load(oxts_path))

        img_depth_path = os.path.join(img_depth_dir, idx_string + '.pt')
        img_depth.append(torch.load(img_depth_path, map_location=torch.device('cpu')))

        img_flow_path = os.path.join(img_flow_dir, idx_string + '.pt')
        img_flow.append(torch.load(img_flow_path, map_location=torch.device('cpu')))

        detection_path = os.path.join(detection_dir, idx_string + '.pt')
        detection.append(torch.load(detection_path, map_location=torch.device('cpu')))
      else:
        idx = idx_sample + i 
        idx_string =  str(idx).zfill(10)

        oxts_path = os.path.join(oxts_dir, idx_string + '.txt')
        oxts_all = np.loadtxt(oxts_path, delimiter=' ')
        oxts.append([oxts_all[k] for k in self.idx_keywords])

        timestamps.append(parser.parse(linecache.getline(timestamp_file, idx+1)))

        img_raw_path = os.path.join(img_raw_dir, idx_string + '.png')
        img_raw.append(self.transform(Image.open(img_raw_path)))     
      

    else:
      return None, None, None, drive, idx_sample
    
    # convert to tensor
      # oxts = torch.tensor(oxts)
    if self.preprocessed:
        oxts = torch.stack(oxts)
        img_depth = torch.stack(img_depth)
        img_flow = torch.stack(img_flow)
        detection = torch.stack(detection)
    else:
        # oxts = torch.tensor(oxts)
        img_raw = torch.stack(img_raw)

    if self.example:
      return oxts, img_raw, timestamps, drive, idx_sample
    elif self.preprocessed: 
      return oxts, img_depth, img_flow, detection
    else:
      return oxts, img_raw


################################### DLAV DATASET ###############################
# USED FOR INFERENCE

class DlavDataset(Dataset):
  def __init__(self, root_dir, scene_name=None, crop_values=None, preprocessed_dataset=False, window_size=2, load_for_example=False):
    # parameters
    self.preprocessed = preprocessed_dataset
    self.window_size = window_size
    self.example = load_for_example
    self.zfill_size=5
    self.crop_values = crop_values
    self.transform  = transforms.Compose([transforms.CenterCrop((360,1224)), transforms.ToTensor()]) # before : 370, 1224
    
    # raw directories
    self.img_raw_dir = root_dir
    self.scene = scene_name

    # preprocessed directories
    self.img_depth_dir = os.path.join(root_dir, 'depth/')
    self.img_flow_dir = os.path.join(root_dir, 'optical_flow/')
    self.detection_dir = os.path.join(root_dir, 'detection/')

    if self.preprocessed==False:
      self.tot_length = len(os.listdir(self.img_raw_dir)) - self.window_size + 1
    else:
      self.tot_length = len(os.listdir(self.img_flow_dir)) - self.window_size + 1


  def __len__(self):
    return self.tot_length
  
  def __getitem__(self, idx):
    idx = idx+1 # image names start at 1
    # get all data in window size
    img_raw = []
    img_depth = []
    img_flow = []
    detection = []
    for i in range(self.window_size):
      if self.preprocessed==False:
        idx_sample = idx + i
        idx_string =  self.scene + '_' + str(idx_sample).zfill(self.zfill_size)
        # print(idx_string)

        img_raw_path = os.path.join(self.img_raw_dir, idx_string + '.png')
        img = Image.open(img_raw_path)
        if self.crop_values == None:
          img = self.transform(img)
        else:
          t=self.crop_values[0]
          l=self.crop_values[1]
          h=self.crop_values[2]
          w=self.crop_values[3]
          img = transforms.functional.to_tensor(img)
          img = transforms.functional.crop(img,t,l,h,w)#100,348,360,1224)
        img_raw.append(img)

      else:
        idx_sample = idx + i + 1
        idx_string = str(idx_sample).zfill(self.zfill_size)

        img_depth_path = os.path.join(self.img_depth_dir, idx_string + '.pt')
        img_depth.append(torch.load(img_depth_path, map_location=torch.device('cpu')))

        img_flow_path = os.path.join(self.img_flow_dir, idx_string + '.pt')
        img_flow.append(torch.load(img_flow_path, map_location=torch.device('cpu')))

        detection_path = os.path.join(self.detection_dir, idx_string + '.pt')
        detection.append(torch.load(detection_path, map_location=torch.device('cpu')))
          
    # convert to tensor
    if self.preprocessed:
      img_depth = torch.stack(img_depth)
      img_flow = torch.stack(img_flow)
      detection = torch.stack(detection)
    else:
      img_raw = torch.stack(img_raw)

    if self.example:
      return img_raw, idx_sample
    elif self.preprocessed: 
      return img_depth, img_flow, detection
    else:
      return img_raw

