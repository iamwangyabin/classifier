# Copy this file to HiFi project, and run it

import os
import csv
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
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
            img = img.cuda()
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

#
#
# def validate(model, loader, set_name, sub_names=None):
#     with torch.no_grad():
#         y_true, y_pred = [], []
#         print("Length of dataset: %d" % (len(loader)))
#         for sample in loader:
#             img, label = sample['image'], sample['label']
#             img = img.cuda()
#             output = model.FENet(img)
#             mask1_fea, mask1_binary, out0, out1, out2, out3 = model.SegNet(output, img)
#             res, prob = one_hot_label_new(out3)
#             # res = level_1_convert(res)[0]
#             y_pred.extend(prob)
#             y_true.extend(label.flatten().tolist())
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     all_results = []
#     for i, sub_task in enumerate(sub_names):
#         mask = (y_true >= i * 2) & (y_true <= 1 + i * 2)
#         idxes = np.where(mask)[0]
#         if len(idxes) == 0:
#             continue
#         ap = average_precision_score(y_true[idxes] % 2, y_pred[idxes])
#         r_acc0, f_acc0, acc0 = calculate_acc(y_true[idxes] % 2, y_pred[idxes], 0.5)
#         best_thres = find_best_threshold(y_true[idxes] % 2, y_pred[idxes])
#         r_acc1, f_acc1, acc1 = calculate_acc(y_true[idxes] % 2, y_pred[idxes], best_thres)
#         all_results.append([set_name, sub_task, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres])
#         print(f"Subtask: {sub_task}\tAP: {ap:.4f},\tAcc0: {acc0:.4f},\tAcc1: {acc1:.4f},\tBestThres: {best_thres:.4f}")
#     return all_results
#
# subfolder_names = {
#     'ForenSynths': ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan", "san", "seeingdark", "stargan",
#                    "stylegan", "stylegan2", "whichfaceisreal"],
#     'DiffusionForensics': ["adm", "ddpm", "diff-stylegan", "if", "midjourney", "projectedgan", "sdv1_new2",
#             "stylegan_official", "dalle2", "diff-projectedgan", "iddpm", "ldm", "pndm", "sdv1_new", "sdv2", "vqdiffusion"],
#     'Ojha': ["dalle", "glide_100_10", "glide_100_27", "glide_50_27", "guided", "ldm_100", "ldm_200", "ldm_200_cfg"],
#     'AntifakePrompt': ['AdvAtk', 'DALLE2', 'Deeperforensics', 'IF', 'lteSR4', 'SD2Inpaint', 'SDXL', 'Backdoor',
#                        'Control', 'DataPoison', 'Lama', 'SD2', 'SD2SuperRes', 'SGXL']
#     }
#
# if __name__ == '__main__':                                                                                                        [782/1913]
#     model = HiFi_Net()
#     print("Model loaded.")
#     # model.eval()
#     # model.cuda()
#     all_results = []
#     for subdata in ['ForenSynths', 'DiffusionForensics', 'AntifakePrompt', 'Ojha']:
#         dataset = load_dataset('nebula/DFBenchmarkPNG', cache_dir='/data/jwang/cache', split=subdata, num_proc=8)
#         dataset.set_format("torch")
#         trans = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#         ])
#
#         def custom_trans(examples):
#             images = []
#             keys = []
#             for image, key in zip(examples["image"], examples["label"]):
#                 image = io.BytesIO(image)
#                 image = Image.open(image)
#                 images.append(trans(image.convert("RGB")))
#                 keys.append(key)
#             examples['image'] = images
#             examples['label'] = keys
#             return examples
#         dataset.set_transform(custom_trans)
#         data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)
#
#         results = validate(model, data_loader, subdata, sub_names=subfolder_names[subdata])
#         all_results.extend(results)
#
#     columns = ['dataset', 'model', 'ap', 'r_acc0', 'f_acc0', 'acc0', 'r_acc1', 'f_acc1', 'acc1', 'best_thres']
#     with open('model_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(columns)
#         for values in all_results:
#             writer.writerow(values)



