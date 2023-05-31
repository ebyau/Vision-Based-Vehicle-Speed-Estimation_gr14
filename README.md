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
To design our model, we have first trained it on a small portion of the dataset (three videos, for approximately 400 images in total) and tried to overfit those data. We started with a neural network that combined CNN layers with FC ones at the end. Nevertheless, it was not able to provide satisfying results. We then switched to a fully convolutional network which was able to overfit the small portion of the dataset. We have also tested our model with and without the detection mask, to see if it really add a benefit. 
All those tests have led to our final model, available in `model.py`. Its associated weights can be loaded from `/weights/weights.pt`.

## Results
### Without Detection Mask
After only 5 epochs, our model has reached a MSE loss of 0.628 on the validation set and 0.895 on the training set. Based on the learning curve, our model is clearly not overfitting.
<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/cb851dac-fe9c-41df-aae6-5fac10cc826b" width="70%">

On the test set, we get a RMSE of 0.786 [m/s].

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/1959e89e-4f24-4831-9c22-3dbb8da0c8c7" width="70%">

### With Detection Mask
<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/9b6e5602-5f3c-4c37-8e37-b26597cec523" width="70%">

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/2847303e-1e77-4075-bb1e-40278f1de28b" width="70%">

## Setup
To setup our model, you first need to get DPT, RAFT and YOLOv8 working. For this purpose, please follow these installation steps :
### Install requirements
`pip install -r requirements.txt`

### Setup Dataset
Place your dataset in the "dataset" folder, using the following data structure:
   
    ├── dataset                 # Test files (alternatively `spec` or `tests`)
    │   ├── file1               # Load and stress tests
    │   │   ├── file1           # Load and stress tests
    │   │   └── unit            # Unit tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests


### Preprocess Data
Our current predictor takes only the pre-processed images as input. It means that it currently doesn't accept raw images, but optical flow, depth and detection data. So the first step to predict speed from a video, is to preprocess it, i.e. extract those features. To do so, run one after the other ;
- `preprocess_depth.py`
- `preprocess_of.py`
- `preprocess_detection.py`

## Train
To train or model your own data, first make sure that the data structure is correct and that the preprocessing has been done properly, and then simply run `train.py`. 

## Test
To test our model on your own data, first make sure that the data structure is correct and that the preprocessing has been done properly, and then simply run `inference.py`. 
 
