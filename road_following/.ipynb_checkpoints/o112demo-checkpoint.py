import torchvision
import torch
import torch.nn as nn 

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
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

model = Net2()
# model = torchvision.models.resnet18(pretrained=False)
# model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
model.num_classes = 2


# model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_steering_model_xycustomCNN3.pth'))
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
    image = PIL.Image.fromarray(image)
    image = transforms.functional.resize(image, (112, 112))

    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


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


speed = 0.20#ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, description='speed gain')
steering_gain_slider = 0.04#ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2, description='steering gain')
steering_dgain_slider = 0 #ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd')
steering_bias_slider = 0#ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias')

# display(speed_gain_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider)


# display(ipywidgets.HBox([y_slider, speed_slider]))
# display(x_slider, steering_slider)


angle = 0.0

import time
def execute(cam):
    global angle
    start_time = time.time()
    image = cam['new']
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    

    
    
    
    angle = np.arctan2(x, y)
    pid = angle * steering_gain_slider 
    # angle_last = angle    
    robot.left_motor.value = max(min(speed + pid, 1.0), 0.0)
    robot.right_motor.value = max(min(speed - pid, 1.0), 0.0)
    timeOneIteration = time.time() - start_time 
    print(timeOneIteration)
    
execute({'new': camera.value})

camera.observe(execute, names='value')