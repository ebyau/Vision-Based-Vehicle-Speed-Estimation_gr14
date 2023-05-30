import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class dlavNet(nn.Module):
    def __init__(self, n_input_channels=3, n_output=1):
        super().__init__()
        ################################################################################
        #                                    LAYERS                                    #
        ################################################################################

        # Input tensors are (360 x 1224) -> Multiple of 8

        # Convolution layers
        self.maxInput = nn.MaxPool2d(kernel_size=(16, 16), stride=8, padding=0)                 # output : 3 x 45 x 153
        self.conv1 = nn.Conv2d(n_input_channels, n_input_channels, kernel_size=7, stride=1)     # output : 3 x 38 x 146
        self.max2 = nn.MaxPool2d(kernel_size=(3, 7), stride=(3, 7), padding=0)                  # output : 3 x 12 x 20
        self.conv2 = nn.Conv2d(n_input_channels, 1, kernel_size=5, stride=1)                    # output : 1 x 8 x 16

        self.max3 = nn.MaxPool2d(kernel_size=(4, 8), stride=(1, 1), padding=0)                 # output : 1 x 5 x 9
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(4, 8), stride=1)                             # output : 1 x 2 x 2
        self.max4 = nn.MaxPool2d(kernel_size=(2, 2))                                           # output : 1 x 1 x 1
    
    def forward(self, x):
        # max -> conv -> leaky ReLU -> max -> conv -> leaky ReLU
        x = self.maxInput(x)                            # In : 3 x 360 x 1224   | Out : 3 x 44 x 152
        x = F.leaky_relu(self.conv1(x))                 # In : 3 x  44 x  152   | Out : 3 x 38 x 146

        x = self.max2(x)                                # In : 3 x  38 x  146   | Out : 3 x 12 x  20
        x = F.leaky_relu(self.conv2(x))                 # In : 3 x  12 x   20   | Out : 1 x  8 x  16

        x = self.max3(x)
        x = F.leaky_relu(self.conv3(x))

        x = self.max4(x)
        return x.view(x.size(0), -1)
       
    
    def predict(self, x):
        self.eval()
        return self.forward(x)

    
