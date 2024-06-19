import io
from io import BytesIO
import cv2
import numbers
import numpy as np
from collections.abc import Sequence
from PIL import ImageFile, Image
from random import random, choice, randint
from scipy.fftpack import dct
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)

def data_augment(img, opt):
    img = np.array(img)
    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)
    return size

class RandomInterpolationResize(torch.nn.Module):
    def __init__(self, size, max_size=None, antialias=None):
        super().__init__()
        self.size = _setup_size(size, error_msg=" (h, w) as size.")
        self.interpolation = [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR,
                              transforms.InterpolationMode.BICUBIC, transforms.InterpolationMode.BOX,
                              transforms.InterpolationMode.HAMMING, transforms.InterpolationMode.LANCZOS,]
        if max_size is not None:
            if not (isinstance(max_size, int) and max_size > 0):
                raise ValueError("max_size must be an integer")
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        interpolation = random.choice(self.interpolation)
        return transforms.functional.resize(img, self.size, interpolation, self.max_size, self.antialias)






class DataAugment:
    def __init__(self, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual):
        self.blur_prob = blur_prob
        self.blur_sig = blur_sig
        self.jpg_prob = jpg_prob
        self.jpg_method = jpg_method
        self.jpg_qual = jpg_qual

    def __call__(self, image):
        image = np.array(image)
        if random() < self.blur_prob:
            sig = sample_continuous(self.blur_sig)
            gaussian_blur(image, sig)

        if random() < self.jpg_prob:
            method = sample_discrete(self.jpg_method)
            qual = sample_discrete(self.jpg_qual)
            image = jpeg_from_key(image, qual, method)

        return Image.fromarray(image)


class DCTTransform:
    def __init__(self, mean_path, var_path, log_scale=True, epsilon=1e-12):
        self.log_scale = log_scale
        self.epsilon = epsilon

        # self.dct_mean = torch.load(mean_path).permute(1, 2, 0).numpy()
        # self.dct_var = torch.load(var_path).permute(1, 2, 0).numpy()
        self.dct_mean = np.load(mean_path)
        self.dct_var = np.load(var_path)

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
        quality_factor = randint(int(self.qf[0]), int(self.qf[1]))
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
