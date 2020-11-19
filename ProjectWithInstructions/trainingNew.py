import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as D
import glob
import PIL.Image
import os
import numpy as np

class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.

        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)

        the convolution layers are meant to handle feature engineering
        the fully connected layer for predicting the steering angle.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


# Define model
print("==> Initialize model ...")
model = NetworkNvidia()
print("==> Initialize model done ...")

def get_x(path):
    """Gets the x value from the image filename"""
    # print(int(path.split('_')[1]))

    return (float(int(path.split('_')[1])) / 50.0)

def get_y(path):
    # s_comma.split(','))
    """Gets the y value from the image filename"""
    return (float(int(path.split('_')[2]) - 50.0) / 50.0)

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))
        
#         if float(np.random.rand(1)) > 0.5:
#             image = transforms.functional.hflip(image)
#             x = -x
#         print("omg")
        image = self.color_jitter(image)
        img_yuv = image.convert('YCbCr')
        img_yuv = transforms.functional.resize(img_yuv, (224, 224))
        img_yuv = transforms.functional.to_tensor(img_yuv)
        img_yuv = img_yuv.numpy()[::-1].copy()
        img_yuv = torch.from_numpy(img_yuv)
        img_yuv = transforms.functional.normalize(img_yuv, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return img_yuv, torch.tensor([x, y]).float()
datasets = []   
datasets.append(XYDataset("Final dataset/test/dataset_xy", random_hflips=False))
# dataset = XYDataset("Final dataset/dataset_xy", random_hflips=False)


for folder in os.listdir("Final dataset/test/Augmentation"):
    # print(folder)
    datasets.append(XYDataset("Final dataset/test/Augmentation/" + folder, random_hflips=False))
            
 # read in our file    if (entry.path.endswith(".jpg")
        
dataset = D.ConcatDataset(datasets)
# dataset = XYDataset("Final dataset/test/dataset_xy", random_hflips=False)

if __name__ == '__main__':


    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )


    model = models.resnet18(pretrained=True)



    model.fc = torch.nn.Sequential(torch.nn.Linear(512, 100),

            torch.nn.Linear(100, 10),
            torch.nn.Linear(10, 2))
    device = torch.device('cuda')
    model = model.to(device)


    NUM_EPOCHS = 120
    BEST_MODEL_PATH = 'best_steering_model_xyResnetMoreLs100.pth'
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())

    for epoch in range(NUM_EPOCHS):
        
        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            test_loss += float(loss)
        test_loss /= len(test_loader)
        
        print('%f, %f' % (train_loss, test_loss))
        if test_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss