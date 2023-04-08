import torch.nn as nn
import torch
from torch import Tensor

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            
        )
        self.fc = nn.Sequential(
            
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

### Source: https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return  out

class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: BasicBlock,
        num_classes: int  = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512*self.expansion, num_classes)
        if num_classes == 1:
            self.fc = nn.Sequential(
                nn.Linear(512*self.expansion, num_classes),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: BasicBlock,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Inherit from nn.Module
class ConvFCNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # Input size: 224 x 224 x 3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 224 x 224 x 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 224 x 224 x 32
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 x 112 x 32
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2), # 56 x 56 x 64,
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 56 x 56 x 64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2), # 28 x 28 x 128,
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 28 x 28 x 128
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 14 x 14 x 128
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2), # 7 x 7 x 256
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2), # 4 x 4 x 256
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=4, stride=4) # 1 x 1 x 256
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=150528, out_features=256),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            
            nn.Linear(in_features=32, out_features=1),
            # End with sigmoid function
            nn.Sigmoid()
        )
        
        self.flatten_image = nn.Flatten()
        
        self.flatten_cnn = nn.Flatten()
    
    # Forward is forward propergation
    def forward(self, x):
        out1 = self.cnn(x)
        
        flatten_x = self.flatten_image(x)
        out2 = self.fc(flatten_x)
        
        # out = out.view(out.size()[0], -1)
        flatten_out1 = self.flatten_cnn(out1)
        
        cat = torch.cat((flatten_out1, out2), dim=1)
        
        out = self.fc2(cat)
        return out


class ConvFCNetv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # Input size: 224 x 224 x 3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 224 x 224 x 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 224 x 224 x 32
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 x 112 x 32
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2), # 56 x 56 x 64,
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 56 x 56 x 64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2), # 28 x 28 x 128,
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 28 x 28 x 128
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 14 x 14 x 128
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2), # 7 x 7 x 256
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2), # 4 x 4 x 256
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0), # 2 x 2 x 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 1 x 1 x 256
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=150528, out_features=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            
            nn.Linear(in_features=32, out_features=1),
            # End with sigmoid function
            nn.Sigmoid()
        )
        
        self.flatten_image = nn.Flatten()
        
        self.flatten_cnn = nn.Flatten()
    
    # Forward is forward propergation
    def forward(self, x):
        out1 = self.cnn(x)
        
        flatten_x = self.flatten_image(x)
        out2 = self.fc(flatten_x)
        
        # out = out.view(out.size()[0], -1)
        flatten_out1 = self.flatten_cnn(out1)
        
        cat = torch.cat((flatten_out1, out2), dim=1)
        
        out = self.fc2(cat)
        return out

class ConvFCNetv3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # Input size: 224 x 224 x 3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 224 x 224 x 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 224 x 224 x 32
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 x 112 x 32
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2), # 56 x 56 x 64,
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 56 x 56 x 64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2), # 28 x 28 x 128,
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 28 x 28 x 128
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 14 x 14 x 128
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2), # 7 x 7 x 256
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2), # 4 x 4 x 256
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0), # 2 x 2 x 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2), # 1 x 1 x 256
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            
            nn.Linear(in_features=32, out_features=1),
            # End with sigmoid function
            nn.Sigmoid()
        )
                
        self.flatten_cnn = nn.Flatten()
    
    # Forward is forward propergation
    def forward(self, x):
        out1 = self.cnn(x)
        
        # out = out.view(out.size()[0], -1)
        flatten_out1 = self.flatten_cnn(out1)
        
        out = self.fc(flatten_out1)
        return out