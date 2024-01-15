import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import ImageFile, Image
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


class BinaryMultiDatasets(Dataset):
    def __init__(self, opt, split='train'):
        self.dataroot = opt.dataroot
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.label_mapping = {'0_real': 0, '1_fake': 1}
        image_extensions = ('.jpg', '.jpeg', '.png')

        for id, subfolder in enumerate(opt.subfolder_names):
            if opt.multicalss_idx[id]:
                classes = os.listdir(os.path.join(self.dataroot, subfolder, split))
            else:
                classes = ['']
            for cls in classes:
                root = os.path.join(self.dataroot, subfolder, split, cls)
                for label in ['0_real', '1_fake']:
                    label_dir = os.path.join(root, label)
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        if img_path.lower().endswith(image_extensions):
                            self.image_pathes.append(img_path)
                            self.labels.append(self.label_mapping[label]+id*2)
        # import pdb;pdb.set_trace()
        if split == 'train':
            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.RandomResizedCrop(opt.cropSize),
                transforms.RandomHorizontalFlip() if opt.random_flip else transforms.Lambda(lambda img: img),
                transforms.Lambda(lambda img: data_augment(img, opt.augment)) if opt.augment else transforms.Lambda(lambda img: img),
                transforms.ToTensor(),
            ]

        else:
            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.CenterCrop(opt.cropSize),
                transforms.ToTensor(),
            ]

        self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform_chain(image)
        return image, label


# class CustomDataset(Dataset):
#     def __init__(self, opt, root):
#         self.opt = opt
#         self.txt_file = opt.txt_file
#         self.root = root
#         self.data = self._load_data()
#         print(len(self.data))
#         if self.opt.isTrain:
#             self.crop_func = transforms.RandomCrop(opt.cropSize)
#         elif self.opt.no_crop:
#             self.crop_func = transforms.Lambda(lambda img: img)
#         else:
#             self.crop_func = transforms.CenterCrop(opt.cropSize)
#
#         if self.opt.isTrain and not self.opt.no_flip:
#             self.flip_func = transforms.RandomHorizontalFlip()
#         else:
#             self.flip_func = transforms.Lambda(lambda img: img)
#
#         if not self.opt.isTrain and self.opt.no_resize:
#             self.rz_func = transforms.Lambda(lambda img: img)
#         else:
#             self.rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
#
#         self.transform = transforms.Compose([
#             self.rz_func,
#             transforms.Lambda(lambda img: data_augment(img, opt)),
#             self.crop_func,
#             self.flip_func,
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         img_path, label = self.data[index]
#         img_path = os.path.join(self.root, img_path)
#         img = Image.open(img_path).convert("RGB")
#         img = self.transform(img)
#         #import pdb;pdb.set_trace()
#         return img, label
#
#     def _load_data(self):
#         data = []
#         with open(self.txt_file, "r") as file:
#             lines = file.readlines()
#             for line in lines:
#                 line = line.strip()
#                 if line:
#                     img_path, label = line.split()
#                     data.append((img_path, int(label)))
#         #import pdb;pdb.set_trace()
#         return data

