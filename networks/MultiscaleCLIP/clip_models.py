from typing import Any, List, Optional, Tuple, Union
from networks.clip import clip
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "RN50x64": 1024,
    "ViT-L/14@336px": 768,
}


class MultiscaleCLIPModel(nn.Module):
    def __init__(self, name, patch_sizes, strides, num_classes=1):
        super(MultiscaleCLIPModel, self).__init__()
        self.model, _ = clip.load(name, device="cpu")
        self.preprocess = CLIPImageProcessor("openai/clip-vit-large-patch14")
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        self.eval()
        self.to_PIL = T.ToPILImage()
        self.to_t = T.ToTensor()
        self.patch_sizes = patch_sizes
        self.strides = strides

    def forward(self, x):
        logits = []
        for ps in self.patch_sizes:
            for s in self.strides:
                logits.append(self.clip_conv(x, ps, s))
        logits = torch.cat(logits, dim=1)
        return logits.mean(1)

    def _extract_patches(self, img, patch_size=32, stride=2) -> torch.Tensor:
        patches = img.unfold(2, patch_size, int(patch_size // stride)).unfold(3, patch_size, int(patch_size // stride)).flatten(2, 3)
        return patches

    def _process_patches(self, patches):
        patches = patches.permute(0, 2, 1, 3, 4).flatten(0, 1)  # B, C, npatch, hp, wp -> B*npatches C h w
        return [self.to_PIL(patch) for patch in patches]
        # return [patch for patch in patches]
        # return patches

    def clip_conv(self, img, patch_size=32, stride=2):
        B, _, h, w = img.shape
        patches = self._extract_patches(img, patch_size, stride)  # B, 3, npatch, hp, wp  (npatch = (hw // patch_size**2))
        patches = self._process_patches(patches)  # List[PIL.Image]  (B*npatch x (3, hp, wp))
        # import pdb;pdb.set_trace()
        logits_per_image = self.infer(patches)  # B*npatch, C
        logits_per_image = logits_per_image.reshape(B, -1)
        return logits_per_image

    @torch.no_grad()
    def infer(self, b_patches):
        """
        infer logits from image patches
        """
        # image_processed = self.preprocess(b_patches).to(self.model.device)
        image_processed = self.preprocess(b_patches, return_tensors="pt").to(self.fc.weight.device)
        features = self.model.encode_image(image_processed['pixel_values'])
        logits_per_image = self.fc(features)
        return logits_per_image


