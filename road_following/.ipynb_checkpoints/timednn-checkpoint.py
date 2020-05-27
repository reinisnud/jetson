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

def get_x(path):
    """Gets the x value from the image filename"""
    return (float(int(path[3:6])) - 50.0) / 50.0

def get_y(path):
    """Gets the y value from the image filename"""
    return (float(int(path[7:10])) - 50.0) / 50.0

def preprocess(image):
    image = PIL.Image.open(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
#     print("success pre")
    return image[None, ...]






def runNetwork(image):
    global angle
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    angle = np.arctan2(x, y)
    print(x, "   ", xy[1])
    
import time
import os

runNetwork(os.path.join('images', 'xy_050_072.jpg'))
total_time = 0
# averagetime = 0
count =0 
for filename in os.listdir('images'):
    name = os.path.join('images', filename)
#     strname = "images\\" + filename
    if os.path.isfile(name):
        start_time = time.time()
        runNetwork(name)
#         print(name)
        print(get_x(filename), get_y(filename))
        count +=1
        
        
        timeOneIteration = time.time() - start_time 
        # print(timeOneIteration, "Seconds")
        total_time += timeOneIteration
        
    else:
        print("No file found")
        
# print(total_time)
print(total_time/count)