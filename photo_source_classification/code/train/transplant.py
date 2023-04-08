import torch
import torch.nn as nn
from classifier import ConvFCNet, ConvFCNetv2, ConvFCNetv3

model1 = torch.load("./results/model_convfcnetv2_epoch10_acc9032")

model2 = ConvFCNetv3()

print(model2)

for idx in range(30):
    layer = model1.cnn.__getattr__("{}".format(idx))
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
        model2.cnn.__getattr__("{}".format(idx)).load_state_dict(layer.state_dict())

# for idx in range(2):
#     layer = model1.fc.__getattr__("{}".format(idx))
#     if isinstance(layer, nn.Linear):
#         model2.fc.__getattr__("{}".format(idx)).load_state_dict(layer.state_dict())

for idx in range(1, 6):
    layer = model1.fc2.__getattr__("{}".format(idx))
    if isinstance(layer, nn.Linear):
        model2.fc.__getattr__("{}".format(idx)).load_state_dict(layer.state_dict())

torch.save(model2, "./transplanted_model_acc9032")