import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from trainer import Trainer
import torch.nn as nn

INPUT_DROPOUT_RATE = 0.5
DROPOUT_RATE = 0.8

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), # [16, 28, 28]
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1), # [16, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2, 0), # [16, 14, 14]
            nn.Dropout2d(INPUT_DROPOUT_RATE),
            
            nn.Conv2d(16, 32, 3, 1, 1), # [32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), # [32, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2, 0), # [32, 7, 7]
            nn.Dropout2d(DROPOUT_RATE),
            
            nn.Conv2d(32, 64, 3, 1, 1), # [64, 7, 7]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), # [64, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2, 0), # [64, 3, 3]
            nn.Dropout2d(DROPOUT_RATE),
        )
        self.fc = nn.Sequential(
            nn.Linear(576, 144),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),
            
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),
            
            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),

            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),

            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),
            
            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),
            
            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),
            
            nn.Linear(72, 10),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root = "./data", train = True, transform = train_transforms, download = True)
val_dataset = datasets.MNIST(root = "./data", train = False, transform = test_transforms, download = True)

batch_size = 128

model = CNN().cuda()
lr = 0.01
num_epoch = 10
optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 1e-5)
loss = nn.CrossEntropyLoss()
trainer = Trainer(model = model, optimizer = optimizer, num_epoch = num_epoch, loss = loss, train_dataset = train_dataset, val_dataset = val_dataset, batch_size = batch_size)
trainer.train()
trainer.validation()

torch.save(model, "./model")