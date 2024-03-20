import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
from networks.NPR.resnet import resnet50

class NPRModel(nn.Module):
    def __init__(self, num_classes=1):
        super(NPRModel, self).__init__()
        self.module = resnet50(num_classes=num_classes)

    def forward(self, x, return_feature=False):
        return self.module(x)

