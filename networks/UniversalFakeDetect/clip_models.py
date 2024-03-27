from networks.clip import clip
import os
import torch
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "RN50x64": 1024,
    "ViT-L/14@336px": 768,
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        # torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, return_feature=False, **kwargs):
        features = self.model.encode_image(x) 
        if return_feature:
            return self.fc(features), features
        return self.fc(features)


class CLIPModel_inc(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel_inc, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.ModuleList([nn.Linear(CHANNELS[name], num_classes) for _ in range(5)])
        for i in range(5):
            state_dict = torch.load(
                os.path.join('./logs/20240130_18_01_23/', 'task_{}.ckpt'.format(i)),
                map_location='cpu')
            self.fc[i].weight.data = state_dict['state_dict']['model.fc.weight']
            self.fc[i].bias.data = state_dict['state_dict']['model.fc.bias']

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        logit = []
        for fc in self.fc:
            logit.append(fc(features))
        # return torch.cat(logit, 1).sum(1)
        return torch.cat(logit, 1).mean(1)
