# Miscellaneous
import matplotlib.pyplot as plt
import numpy as np
import time, sys, torch

# torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import datasets, transforms

# Custom
from model import dlavNet
from dataset import KittiDataset

# Use GPU/CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using {} device..".format(device))

# Dataset directory
dataset_dir = '/dataset_kitti'

# Create model
net = dlavNet()
net.to(device)

# Training Parameters
mask = True                                         # Decide wheter or not we zeroe region where we detect vehicles
epochs = 10                                         # Number of epochs to train for
batch_size = 20                                     # Number of samples per batch
split_ratio = [60, 20, 20]                          # train, validation and test ratio
lr = 3e-3                                           # Learning Rate

# Define optimizer and criterion
optimizer = optim.Adam(net.parameters(), lr=lr)     # Optimizer
criterion = nn.MSELoss()                            # Loss Function

# Create train / validation set and loader
train_set = KittiDataset(dataset_dir, mode='train', split_ratio=split_ratio, window_size=1, device=device, preprocessed_dataset=True)
val_set = KittiDataset(dataset_dir, mode='val', split_ratio=split_ratio, window_size=1, device=device, preprocessed_dataset=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

# Initialize monitoring data
batch = 0
steps = 0
running_loss = 0
print_every = 20

######################################## TRAINING ####################################

for e in range(epochs):
    start = time.time()
    batch = 0
    for oxts, img_depth, img_flow, detection in iter(train_loader):
        batch += 1
        print("batch {} with {} data".format(batch, batch*batch_size))
        with torch.no_grad():
            img_flow = img_flow.squeeze(1) # Remove dimension 1 of the tensor
            masked_of = torch.zeros_like(img_flow) # Create an array to store masked optical flow
            targets = torch.zeros((img_flow.shape[0], 1), device=device) # Create array to store targets

            # Apply mask to the optical flow
            for i in range(img_flow.shape[0]):
                targets[i] = torch.linalg.norm(oxts[i, 0, 1, :])
                if mask:
                    masked_of[i] = detection[i] * img_flow[i] # Element-wise multiplication -> Region where vehicle is detected are zeroed in the optical flow

            if mask:
                masked_of = masked_of.squeeze(1)
                input = torch.cat((img_depth, masked_of), dim=1)
            else:
                input = torch.cat((img_depth, img_flow), dim=1)

        steps += 1

        net.train()           # Set the NN in "train" mode
        optimizer.zero_grad() # Reset the gradient 
        output = net(input)  # Forward propagation through the NN
        
        loss = criterion(output, targets) # Compute the loss

        loss.backward()  # Backward propagation to get the gradient
        optimizer.step() # Run one optimization step
        
        running_loss += np.sqrt(loss.item())
        
        if steps % print_every == 0:
            accuracy = 0
            with torch.no_grad():
                stop = time.time()
                # Test accuracy
                net.eval() 
                for ii, (oxts, img_depth, img_flow, detection) in enumerate(val_loader):                                                       # Set the NN in "eval" mode -> useful for batch norm & dropout
                    img_flow = img_flow.squeeze(1) # Remove dimension 1 of the tensor -> May be done when saving
                    masked_of = torch.zeros_like(img_flow) # Create an array to store masked optical flow
                    targets = torch.zeros((img_flow.shape[0], 1), device=device) # Create array to store targets

                    # Apply mask to the optical flow
                    for i in range(img_flow.shape[0]):
                        targets[i] = torch.linalg.norm(oxts[i, 0, 1, :]) # Compute norm of the speed
                        if mask:
                            masked_of[i] = detection[i] * img_flow[i] # Element-wise multiplication -> Region where vehicle is detected are zeroed in the optical flow
                
                    if mask:
                        masked_of = masked_of.squeeze(1)
                        input = torch.cat((img_depth, masked_of), dim=1)
                    else:
                        input = torch.cat((img_depth, img_flow), dim=1)

                    output = net.predict(input)                                                # Forward pass + last activation
                    accuracy += criterion(output, targets)                                     # Compute MSE


            
            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Loss: {:.4f}..".format(running_loss/print_every),
                  "Test accuracy: {:.4f}..".format(accuracy/(ii+1)),
                  "{:.4f} s/batch".format((stop - start)/print_every)
                 )
            running_loss = 0
            start = time.time()









