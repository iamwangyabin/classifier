import os
import cv2
import io
import json
import warnings
import numpy as np
import torchvision.transforms as transforms
from random import random, choice, randint
from io import BytesIO
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import dct
import torch
from torch.utils.data import Dataset
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

class DCTTransform:
    def __init__(self, log_scale=True, epsilon=1e-12):
        self.log_scale = log_scale
        self.epsilon = epsilon

        self.dct_mean = torch.load('./networks/weights/dct_mean').permute(1, 2, 0).numpy()
        self.dct_var = torch.load('./networks/weights/dct_var').permute(1, 2, 0).numpy()


    def __call__(self, image):
        image = np.array(image)
        image = dct(image, type=2, norm="ortho", axis=0)
        image = dct(image, type=2, norm="ortho", axis=1)
        # log scale
        if self.log_scale:
            image = np.abs(image)
            image += self.epsilon  # no zero in log
            image = np.log(image)
        # normalize
        image = (image - self.dct_mean) / np.sqrt(self.dct_var)
        image = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float)
        return image


class RandomCompress:
    def __init__(self, method="JPEG", qf=[60, 100]):
        self.qf = qf
        self.method = method

    def __call__(self, image):
        outputIoStream = io.BytesIO()
        quality_factor = randint(self.qf[0], self.qf[1])
        image.save(outputIoStream, self.method, quality=quality_factor, optimize=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)



class Compress:
    def __init__(self, method="JPEG", qf=100):
        self.qf = qf
        self.method = method

    def __call__(self, image):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, self.method, quality=self.qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)


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
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

        self.transform_chain = transforms.Compose(opt.trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform_chain(image)
        return image, label

