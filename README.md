# Ego-vehicle speed estimation from a monocular RGB camera.
This repository implements a deep learning based ego-vehicle speed estimator using three sub-modules : A depth estimator ([DPT](https://github.com/isl-org/DPT)), an optical flow ([RAFT](https://github.com/princeton-vl/RAFT)) and a detector ([YOLOv8](https://github.com/ultralytics/ultralytics)). 

## Dataset
We trained our model on RGB videos from thtree categories of the ([KITTI-Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php)) dataset : road, residential and city. The three categories are usefull for having a wide variety of velocities, traffic load as well as multiple scene. The videos have been randomly selecte to create three datasets (training, validation and testing). Each dataset contains frames of road, residential and city categories, and their percentage is relative to the overall numbers of frames per category. Each frame of a video can only be seen on one of the dataset, so that there is no data leakage across training, validation and testing.

## Method
Our model takes as input RGB videos and feed it to an optical flow estimator, a depth estimator and a detector. We test two setup. In the first one "with mask", the detector is used to find regions in the image where elements (such as cars, or tramways) that could perturbate the speed estimation are. For example, a car in front of us coming in the opposite direction might cause the model think that speed is higher that what it actually is, as shown in [this paper](https://arxiv.org/pdf/1907.06989.pdf). The detection is used to zero those corresponding regions in the optical flow and depth predictions. Then, these predictions are concatenated (optical flow from two images, depth from the most recent image in time) within a single tensor and feeded into our neural network.
The second one doesn't use the detection and simply concatenate the depth estimation with the optical flow estimation within a tensor and feed it into the neural network. This latter has the same structure for each setup and is simply composed of sequences of "max-pooling -> Leaky ReLU -> 2D Convolution" without any fully connected layer in it. For the "mask" setup the use of max pooling layers is conceptually important because this is what is supposed to remove the regions of the images that have been zeroed. The output of the NN is the norm of the ego-vehicle speed. 

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/654b8ab7-8e50-489b-931c-4a3ac0b17efe" width="60%">

The best parameters obtained for the training are the following:

| Parameter  | Value |
| ------------- | ------------- |
| Optimizer  | Adam  |
| Learning rate | 3e-3 |
| Epoch | 20 |
| batch size | 20 |

## Test Conducted
To design our model, we have first trained it on a small portion of the dataset (three videos, for approximately 400 images in total) and tried to overfit those data. We started with a neural network that combined CNN layers with FC ones at the end. Nevertheless, it was not able to provide satisfying results. We then switched to a custom convolutional network which was able to overfit the small portion of the dataset. We have also tested our model with and without the detection mask, to see if it really add a benefit. 
All those tests have led to our final model, available in `model.py`. Its associated weights can be loaded from `/weights/weights.pt`.

## Results
### Without Detection Mask
Our model has reached a MSE loss of 0.628 on the validation set and 0.895 on the training set. Based on the learning curve, our model is clearly not overfitting. The fact that the validation loss is less than the training loss may come from the fact that the validation loss is computed after having updated the weights. On the test set, we get a RMSE of 1.077 [m/s].

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/cb851dac-fe9c-41df-aae6-5fac10cc826b" width="60%">

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/328c5468-729e-4a65-b2d3-cc191c11dda3" width="60%">

### With Detection Mask
The model with a detection mask has a higher validation loss but is better on the training set, probably because the detection mask act as a sort of Dropout on the input. We report a RMSE of 0.882 [m/s] on the test set which is, for the best of our knowledge, better on the Kitti dataset than the actual state of the art methods. 

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/9b6e5602-5f3c-4c37-8e37-b26597cec523" width="60%">

<img src="https://github.com/stiefen1/monocam-to-ego-vehicle-speed/assets/78551150/118728d1-218c-48aa-903d-da4a4fc05295" width="60%">

## Conclusion
Our network, inspired mainly by the ([paper]((https://arxiv.org/pdf/1802.07094.pdf)) of Robert-Adrian Rill is freely downloadable. Four major concepts should be raised.
- We introduced a detection mask for blocking perturbation of the optical flow. This helped to reduced the error, especially at low speed where motion of of other vehicles have a greater impact on the optical flow.
- The network is modular, in the sens that each sub-modules can be independently changed. For future work, if existing at time of the reading, a more up to date network can be used. Another possibility would be to use a custom network (for example, made by other students) for predicting the depth.
-  The velocities of the other vehicle is not implemented yet, however, it could be implemented by inverting the detection mask and predicting their ground truth. This is something which has been done in ([this paper](https://arxiv.org/abs/1802.07094)).
- As a downside, our network is sensible to the sampling frequency of the camera. Our trained model use a sampling frequency of 10Hz, the same as the KITTI-Raw dataset. In addtion, the setup of the camera is important too, for having a sufficient front road visualization. By augmenting the dataset, a more robust model could be learned, for example for having a different point of view. For those reasons, the inference test is probably not as accurate as our testing set. Concerning the sampling frequency, it could be only a scaling factor.

Overall, our model performed great results with a smaller RMSE than the original paper. However, they used a bigger dataset than us and further experiments and training should still be computed for validation.

## Setup
To setup our model, you first need to get DPT, RAFT and YOLOv8 and their dependencies. For this purpose, please follow these installation steps :
### Install requirements
`pip install -r requirements.txt`

### Setup Dataset
To download the KITTI dataset, the original ([script](https://github.com/Deepak3994/Kitti-Dataset)) has been modified in order to separate the categories and keeping only the relative data (RGB images from camera 2 and oxts file where the ego-vehicle speed is recorded). The `raw_data_downloader.sh` can be run to extract the dataset, with its first argument the path to saving directory and second argument for the temporary directory used for unzipping and deleting non-usefull data.


### Preprocess Data
Our current predictor takes only the pre-processed images as input. It means that it currently doesn't accept raw images, but optical flow, depth and detection data. So the first step to predict speed from a video, is to preprocess it, i.e. extract those features. To do so, run one after the other ;
- `preprocess_depth.py`
- `preprocess_of.py`
- `preprocess_detection.py`

## Train
To train or model your own data, first make sure that the data structure is correct and that the preprocessing has been done properly, and then simply run `train.py`. 

## Test
To test our model on your own data, first make sure that the data structure is correct and that the preprocessing has been done properly, and then simply run `inference.py`. 
