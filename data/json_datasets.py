import os
import cv2
import json
import warnings
import torchvision.transforms as transforms
import albumentations as A

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import ImageFile, Image


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

def check_transform_lib(transform):
    module_name = transform.__class__.__module__
    if module_name.startswith('albumentations'):
        return 'albumentations'
    elif module_name.startswith('torchvision'):
        return 'torchvision'
    else:
        return 'unknown'

class BinaryJsonDatasets(Dataset):
    def __init__(self, opt, data_root, subset='all', split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            # if os.path.exists(img_full_path):
            self.image_pathes.append(img_full_path)
            self.labels.append(label)
            # else:
            #     print('pass, not exit {}'.format(img_full_path) )

        for idx, transform in enumerate(opt.trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(opt.trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(opt.trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = cv2.imread(img_path) # this is BGR, different from PIL
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = Image.open(img_path).convert('RGB')
            image = self.transform_chain(image)
        return image, label

