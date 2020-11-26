import torchvision
import torch
import torch.nn as nn 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(
                                      nn.Conv2d(in_channels=3,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1), 
                                     nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                    nn.Conv2d(in_channels=64,
                                                out_channels=128,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1), 
                                      nn.ReLU(inplace=True),        
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(in_channels=128,
                                                out_channels=256,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                       nn.Conv2d(in_channels=256,
                                                out_channels=512,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
#                                        nn.Conv2d(in_channels=512,
#                                                 out_channels=512,
#                                                 kernel_size=3,
#                                                 stride=1,
#                                                 padding=1), 
#                                       nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2))
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(4608, 2))
    def forward(self, x):
      
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 4608)
        
        # Classify the images
        x = self.classifier(x)

        return x

# model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
# lastlayer = torch.nn.Linear(1000, 2)
# model = torch.nn.Sequential(model, lastlayer)

# model = Net()
# # model = torchvision.models.resnet18(pretrained=False)
# # model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
# model.num_classes = 2


# # model.fc = torch.nn.Linear(512, 2)torch.jit.load()
# model.load_state_dict(torch.jit.load('best_steering_model_xynewCNN112.pth'))
# device = torch.device('cuda')
# model = model.to(device)
# print(model.eval().half())

import torch
from torch2trt import TRTModule
device = torch.device('cuda')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('best_steering_model_xyResnet100.10.1V2_trt.pth'))

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.resize(image, (224, 224))
    img_yuv = image.convert('YCbCr')

    img_yuv = transforms.functional.to_tensor(img_yuv).to(device).half()
    img_yuv.sub_(mean[:, None, None]).div_(std[:, None, None])
    return img_yuv[None, ...]


from IPython.display import display
import ipywidgets
import traitlets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera()

# image_widget = ipywidgets.Image()

# traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)

# display(image_widget)

from jetbot import Robot

robot = Robot()


speed = 0.16
steering_gain_slider = 0.03
steering_dgain_slider = 0 
steering_bias_slider = 0
y=75

def calculateAngle(x):#,y):
    global y
    fPivYLimit = 80

    # TEMP VARIABLES
    nMotPremixL = 0 # Motor (left)  premixed output        (-128..+127)
    nMotPremixR = 0 # Motor (right) premixed output        (-128..+127)
    nPivSpeed = 0   # Pivot Speed                          (-128..+127)
    fPivScale = 0.0 # Balance scale b/w drive and pivot    (   0..1   )
    # Calculate Drive Turn output due to Joystick X input
    if(y>=50):
      # Forward
        nMotPremixL = 100.0 if x>=0 else 100.0 + x
        nMotPremixR = 100.0 - x if x>=0 else 100.0
    else:
        # Reverse
        nMotPremixL = 100.0 - x if x>=0 else 100.0
        nMotPremixR = 100.0 if x>=0 else 100.0 + x

    # Scale Drive output due to Joystick Y input (throttle)
    nMotPremixL = nMotPremixL * y/100.0
    nMotPremixR = nMotPremixR * y/100.0

    # Now calculate pivot amount
    #  - Strength of pivot (nPivSpeed) based on Joystick X input
    #  - Blending of pivot vs drive (fPivScale) based on Joystick Y input
    nPivSpeed = x
    fPivScale = 0.0 if abs(y)>fPivYLimit else (1.0-abs(y)/fPivYLimit)

    # Calculate final mix of Drive and Pivot
    nMotMixL = int((1.0-fPivScale)*nMotPremixL + fPivScale*( nPivSpeed)) # Motor (left)  mixed output           (-128..+127)
    nMotMixR = int((1.0-fPivScale)*nMotPremixR + fPivScale*(-nPivSpeed)) # Motor (right) mixed output           (-128..+127)
    
    robot.left_motor.value = nMotMixL * 0.00188
    robot.right_motor.value = nMotMixR * 0.0020

angle = 0.0

import time
def execute(cam):
    global angle
    start_time = time.time()
    image = cam['new']
    xy = model_trt(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0] * 50
#     y = xy[1] * 50 + 50
    calculateAngle(x)
    print("x: ", xy[0])
#     print("y: ", xy[1])
    
    
    
#     angle = np.arctan2(x, y)
#     pid = angle * steering_gain_slider 
#     # angle_last = angle    
#     robot.left_motor.value = max(min(speed + pid, 1.0), 0.0)
#     robot.right_motor.value = max(min(speed - pid, 1.0), 0.0)
#     timeOneIteration = time.time() - start_time
#     print("Pirmais", xy[0])
#     print(xy[1])
#     print(y)
#     print(angle)
#     print("PID", pid)
#     print(timeOneIteration)
    
execute({'new': camera.value})

camera.observe(execute, names='value')


time.sleep(240)

camera.unobserve(execute, names='value')
time.sleep(0.1)
robot.stop()