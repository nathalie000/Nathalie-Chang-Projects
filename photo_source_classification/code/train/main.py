import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from cv2 import imread

from trainer import Trainer
from img_dataset import ImgDataset

# Inherit from nn.Module
class BrandClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # Input size: 224 x 224 x 3
            
        )
        self.fc = nn.Sequential(
            # End with sigmoid function
            nn.Sigmoid()
        )
    
    # Forward is forward propergation
    def forward(self, x):
        out = self.cnn(x)
        # Serve as Flatten in Homework 4
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

# Read training dataset
train_x = np.load("./data_trunk/train_x_trunk0.npy")
train_y = np.load("./data_trunk/train_y_trunk0.npy")

print(type(train_x), train_x.shape)
print(type(train_y), train_y.shape)

# Read training dataset
val_x = np.load("./data_trunk/test_x_trunk0.npy")
val_y = np.load("./data_trunk/test_y_trunk0.npy")

print(type(val_x), val_x.shape)
print(type(val_y), val_y.shape)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # Add any data argumentation here
    # e.g:
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    # transforms.RandomResizedCrop(),
    # transforms.Normalize(mean, std)
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# # Read training dataset
# train_x, train_y = dataset_reader(train = True)
# # Read validation dataset
# val_x, val_y = dataset_reader(train = False)

# Wrap the x and y using ImgDataset
train_dataset = ImgDataset(train_x, train_y, train_transforms)
val_dataset = ImgDataset(val_x, val_y, test_transforms)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

model = torchvision.models.resnet18(num_classes=1)
num_epoch = 10
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)
loss = nn.CrossEntropyLoss()
batch_size = 64

trainer = Trainer(model, optimizer, num_epoch, loss, train_dataset, val_dataset, batch_size, device)

trainer.train()
trainer.validation()