import os
import cv2
import random
import json
import warnings


import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations import Normalize
from data.albu_aug import FrequencyPatterns

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
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = Image.open(img_path).convert('RGB')
            image = self.transform_chain(image)
        return image, label



class BinaryJsonRealMimcDatasets(Dataset):
    # the same as BinaryJsonDatasets, but use albu_aug to make real samples as fake samples
    # only support albumentations transforms
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
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

        self.pollution_probability = opt.pollution_probability

        for idx, transform in enumerate(opt.trsf):
            self.lib = check_transform_lib(transform)
        import pdb;pdb.set_trace()

        if self.lib == 'albumentations':
            normalize_index = next((i for i, t in enumerate(opt.trsf) if isinstance(t, Normalize)), -1)
            self.transform_chain = A.Compose(opt.trsf[:normalize_index])
            self.transform_real2fake = A.Compose([FrequencyPatterns(p=1)])
            self.transform_norm = A.Compose(opt.trsf[normalize_index:])
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(opt.trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            image = self.transform_chain(image=image)["image"]
            if label == 0:
                if random.random() <= self.pollution_probability:
                    image = self.transform_real2fake(image=image)['image']
                    label = 1
            image = self.transform_norm(image=image)["image"].float()

        elif self.lib == 'torchvision':
            image = Image.open(img_path).convert('RGB')
            image = self.transform_chain(image)
        return image, label


