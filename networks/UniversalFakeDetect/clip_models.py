from networks.clip import clip
from PIL import Image
import torch
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        # torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)

