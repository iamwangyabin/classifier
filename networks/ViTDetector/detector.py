import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
import timm



class ViTModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ViTModel, self).__init__()

        self.backbones = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.backbones[name]['embedding_size'], num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, return_feature=False):
        with torch.no_grad():
            features = self.backbone.forward_features(x)['x_norm_clstoken']
        if return_feature:
            return features
        return self.fc(features)

