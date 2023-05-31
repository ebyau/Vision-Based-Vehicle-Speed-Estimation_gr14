# Ego-vehicle speed estimation from a monocular RGB camera.
This repository implements a deep learning based ego-vehicle speed estimator using three sub-modules : A depth estimator (DPT), an optical flow (RAFT) and a detector (YOLOv8). 

## Dataset
We trained our model on RGB videos from thtree categories of the Kitti dataset : road, residential and city.  EXPLAIN HOW WE SEPARATE DATA BETWEEN TRAIN VAL TEST

## Method
Our model takes as input RGB videos and feed it to an optical flow estimator, a depth estimator and a detector. The detector is used to find regions in the image where elements (such as cars, or tramways) that could perturbate the speed estimation are. For example, a car in front of us coming in the opposite direction might cause the model think that speed is higher that what it actually is, as shown in [this paper](https://arxiv.org/pdf/1907.06989.pdf).
The detection is used to zeroe those corresponding regions in the optical flow and depth predictions. Then, these predictions are concatenated (optical flow from two images, depth from the most recent image in time) within a single tensor and feeded into our convolutional neural network. This latter is simply composed of sequences of "max-pooling -> Leaky ReLU -> 2D Convolution" without any fully connected layer in it. The use of max pooling layer is conceptually important because this is what is supposed to remove the regions of the images that have been zeroed. The output of the NN is the norm of the ego-vehicle speed. 

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/654b8ab7-8e50-489b-931c-4a3ac0b17efe" width="70%">

The best parameters obtained for the training are the following:

| Parameter  | Value |
| ------------- | ------------- |
| Optimizer  | Adam  |
| Learning rate | 3e-3 |
| Epoch | 20 |
| batch size | 20 |

## Test Conducted
To design our neural network, we have first trained it on a small portion of the dataset (three videos, for approximately 400 images in total) and tried to overfit those data. We tested our model with and without the detection mask, to see if it really has a benefit. 

## Results

## Setup
To setup our model, you first need to get DPT, RAFT and YOLOv8 working. For this purpose, please follow these installation steps :
### Install requirements
`pip install -r requirements.txt`

### 

## Test
 
