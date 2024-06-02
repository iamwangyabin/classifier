import torch
import torch.nn as nn
import timm

class TIMMModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(TIMMModel, self).__init__()
        self.backbone = timm.create_model(name, pretrained=True)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.backbone.num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, return_feature=False):
        with torch.no_grad():
            if self.backbone.attn_pool is not None:
                features = self.backbone.attn_pool(self.backbone.forward_features(x))
            elif self.backbone.global_pool == 'avg':
                features = self.backbone.forward_features(x)[:, self.backbone.num_prefix_tokens:].mean(dim=1)
            elif self.backbone.global_pool:
                features = self.backbone.forward_features(x)[:, 0]  # class token

        if return_feature:
            return self.fc(features), features

        return self.fc(features)

