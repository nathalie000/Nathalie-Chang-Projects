import torch
from torch import nn
import torch.nn.utils.prune as prune
import torchvision.models as models
from torchinfo import summary


class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.regnet_x_1_6gf(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 525) # predict 525 classes

    def forward(self, img_b):
        cls_b = self.model(img_b)
        return cls_b
    
class StudentNet(nn.Module):
    def __init__(self, prune=False):
        super().__init__()

        self.model = models.shufflenet_v2_x0_5(weights='DEFAULT') # small
        conv5_out = 64 # origin: 1024
        self.model.conv5 = nn.Sequential(
            nn.Conv2d(192, conv5_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(conv5_out),
            nn.ReLU(inplace=True),
        )
        self.model.fc = nn.Linear(conv5_out, 525) # predict 525 classes
        if prune:
            self.pruning(0)

    def forward(self, img_b):
        cls_b = self.model(img_b)
        return cls_b
    
    def all_modules(self):
        modules = ['conv1', 'conv5', 'fc']
        for i in range(1, 3):
            for j in range(4):
                modules.append(f'stage2.{j}.branch{i}')
                modules.append(f'stage4.{j}.branch{i}')
            for j in range(8):
                modules.append(f'stage3.{j}.branch{i}')
        return modules

    def pruning(self, w=0.6):
        def prune_module(module):
            parameters_to_prune = ()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune += ((module, 'weight'), )
            elif isinstance(module, torch.nn.Sequential):
                for submodule in module.children():
                    parameters_to_prune += prune_module(submodule)
            return parameters_to_prune

        modules_to_prune = self.all_modules()
            
        parameters_to_prune = ()
        for name, module in self.model.named_modules():
            if name in modules_to_prune:
                parameters_to_prune += prune_module(module)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=w,
        )

    def remove_pruning(self):
        def remove_module(module):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.remove(module, 'weight')
            elif isinstance(module, torch.nn.Sequential):
                for submodule in module.children():
                    remove_module(submodule)

        modules_to_prune = self.all_modules()
            
        for name, module in self.model.named_modules():
            if name in modules_to_prune:
                remove_module(module)


if __name__ == '__main__':
    model = TeacherNet()
    summary(model, input_size=(1, 3, 224, 224))

    model = StudentNet()
    summary(model, input_size=(1, 3, 224, 224))