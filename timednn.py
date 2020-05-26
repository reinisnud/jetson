import torchvision
import torch

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_steering_model_xyRes18.pth'))
device = torch.device('cuda')
model = model.to(device)
print(model.eval().half())


import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.open(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    print("success pre")
    return image[None, ...]






def runNetwork(image):
    global angle
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    angle = np.arctan2(x, y)
    print(angle)

import os
for filename in os.listdir('images'):
	if os.path.isfile(filename):
    	runNetwork('images/' + filename)