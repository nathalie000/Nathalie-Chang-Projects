import torch
import torch.nn as nn
from classifier import ConvFCNetv2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Ablation test on ConvFCNetv2: 89.76%
model = torch.load("./results/model_convfcnetv2_epoch10_acc9032").cpu()

model_after = ConvFCNetv2().cpu()

print(model)

model_after.cnn.load_state_dict(model.cnn.state_dict())
model_after.fc2.load_state_dict(model.fc2.state_dict())

for idx in range(6):
    layer = model.fc.__getattr__("{}".format(idx))
    if isinstance(layer, nn.Linear):
        st_dict = layer.state_dict()
        st_dict["weight"] = torch.zero_(st_dict["weight"])
        st_dict["bias"] = torch.zero_(st_dict["bias"])
        model_after.fc.__getattr__("{}".format(idx)).load_state_dict(st_dict)
        assert(torch.eq(model_after.fc.__getattr__("{}".format(idx)).weight, torch.zero_(st_dict["weight"])).all())
        assert(torch.eq(model_after.fc.__getattr__("{}".format(idx)).bias, torch.zero_(st_dict["bias"])).all())

torch.save(model_after, "./ablation_model_acc9032")
print("Number of parameters: {}".format(count_parameters(model)))
print("Number of parameters: {}".format(count_parameters(model.cnn)))
print("Number of parameters: {}".format(count_parameters(model.fc2)))