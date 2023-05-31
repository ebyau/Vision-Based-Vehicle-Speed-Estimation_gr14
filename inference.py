from torch.utils.data import DataLoader, Dataset
from dataset import DlavDataset
from torchvision import transforms
import os
from PIL import Image
import datetime
from dateutil import parser
import linecache
import json

USE_GPU = False

path_to_project = ''
dlav_set_dir = '/Project/dlav_dataset/Test5'
batch_size=1

dlav_set = DlavDataset(root_dir = dlav_set_dir, preprocessed_dataset=True, window_size=1)
dlav_loader = DataLoader(dlav_set, batch_size=batch_size, shuffle=False)

# Check device
device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")
path_to_weights = path_to_project + '/weights/weights_with_mask_01.pt'
net = dlavNet()
net.load_state_dict(torch.load(path_to_weights))
net.to(device)
net.eval() 

pred_vec = np.array([])

output_json = {"project":"project_name", "output":[]}

batch = 0
steps = 0
print_every = 20
use_mask = True

start = time.time()
batch = 0
with torch.no_grad():
  for ii, (img_depth, img_flow, detection) in enumerate(dlav_loader):
    batch += 1
    img_flow = img_flow.squeeze(1) # Remove dimension 1 of the tensor -> May be done when saving
    targets = torch.zeros((img_flow.shape[0], 1), device=device) # Create array to store targets
    if use_mask:
      masked_of = torch.zeros_like(img_flow) # Create an array to store masked optical flow

    # Apply mask to the optical flow
    for i in range(img_flow.shape[0]):
      if use_mask:
        masked_of[i] = detection[i] * img_flow[i] # Element-wise multiplication -> Region where vehicle is detected are zeroed in the optical flow
      
    if use_mask:    
      masked_of = masked_of.squeeze(1)
      net_input = torch.cat((img_depth, masked_of), dim=1)
    else:  
      net_input = torch.cat((img_depth, img_flow), dim=1)

    output = net(net_input)  # Forward propagation through the NN
    pred_vec = np.append(pred_vec, output.item())

    output_json["output"].append({"frame":ii, "predictions":[round(output.item(), 3)]})

    print("batch {} with {} data | Sample prediction : {:.2f} m/s".format(batch, batch*batch_size, output[0].item()))

with open(path_to_project + 'json_data_test5.json', 'w') as outfile:
    json.dump(output_json, outfile)
