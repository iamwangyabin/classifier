import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
import timm

class ViTModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ViTModel, self).__init__()
        self.backbone = timm.create_model(name, pretrained=True)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, return_feature=False):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
        if return_feature:
            return self.fc(features), features
        return self.fc(features)

