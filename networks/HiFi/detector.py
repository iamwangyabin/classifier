# Copy this file to HiFi project, and run it

import io
import os
import csv
import cv2
import json
import numpy as np
from io import BytesIO
from random import random, choice
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from PIL import ImageFile, Image
from omegaconf import OmegaConf, ListConfig
from typing import Tuple, List, Iterable, Any
import hydra
import argparse

import torchvision.transforms as transforms
import torch
import torch.utils.data
from torch.utils.data import Dataset

from utils.utils import *
from HiFi_Net import HiFi_Net

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def remove_config_undefined(cfg):
    itr: Iterable[Any] = range(len(cfg)) if isinstance(cfg, ListConfig) else cfg

    undefined_keys = []
    for key in itr:
        if cfg._get_child(key) == '---':
            undefined_keys.append(key)
        elif OmegaConf.is_config(cfg[key]):
            remove_config_undefined(cfg[key])
    for key in undefined_keys:
        del cfg[key]
    return cfg

def load_config(path, remove_undefined=True):
    cfg = OmegaConf.load(path)
    if '_base_' in cfg:
        for base in cfg['_base_']:
            cfg = OmegaConf.merge(load_config(base, remove_undefined=False), cfg)
        del cfg['_base_']
    if remove_undefined:
        cfg = remove_config_undefined(cfg)
    return cfg

def load_config_with_cli(path, args_list=None, remove_undefined=True):
    cfg = load_config(path, remove_undefined=False)
    cfg_cli = OmegaConf.from_cli(args_list)
    cfg = OmegaConf.merge(cfg, cfg_cli)
    if remove_undefined:
        cfg = remove_config_undefined(cfg)
    return cfg


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


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"
    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def validate(model, loader):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            output = model.FENet(img)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = model.SegNet(output, img)
            res, prob = one_hot_label_new(out3)
            y_pred.extend(prob)
            y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, num_real, num_fake


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = HiFi_Net()
    print("Model loaded.")

    all_results = []

    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                                      num_workers=conf.datasets.loader_workers)
            ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, num_real, num_fake = validate(model, data_loader)
            print(f"{set_name} {subset}")
            print(f"AP: {ap:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([set_name, subset, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres,
                                num_real, num_fake])

    columns = ['dataset', 'sub_set', 'ap', 'r_acc0', 'f_acc0', 'acc0', 'r_acc1', 'f_acc1', 'acc1', 'best_thres',
               'num_real', 'num_fake']
    with open('model_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)



