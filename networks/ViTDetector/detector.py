import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models


dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}


class ViTModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ViTModel, self).__init__()
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[name]['name'])
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

