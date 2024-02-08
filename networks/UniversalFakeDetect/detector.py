import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from PIL import Image
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
import shutil
from datasets import load_dataset

from networks.UniversalFakeDetect.clip_models import CLIPModel


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


def validate(model, loader, find_thres=False):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        for sample in loader:
            img, label = sample['image'], sample['label']
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    import pdb;pdb.set_trace()
    # for i, sub_task in enumerate(self.opt.dataset.val.subfolder_names):
    #     mask = (all_gts >= i * 2) & (all_gts <= 1 + i * 2)
    #     idxes = np.where(mask)[0]
    #     if len(idxes) == 0:
    #         continue
    #     acc, ap = validate(all_gts[idxes] % 2, all_preds[idxes])[:2]

    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres



if __name__ == '__main__':
    model = CLIPModel('ViT-L/14')
    state_dict = torch.load('networks/UniversalFakeDetect/fc_weights.pth', map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print("Model loaded..")
    model.eval()
    model.cuda()
    for subdata in ['ForenSynths', 'DiffusionForensics', 'AntifakePrompt', 'Ojha']:
        dataset = load_dataset('nebula/DFBenchmark', cache_dir='/data/jwang/cache', split=subdata, num_proc=8)
        dataset.set_format("torch")
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        def custom_trans(examples):
            images = []
            keys = []
            for image, key in zip(examples["image"], examples["label"]):
                images.append(trans(image.convert("RGB")))
                keys.append(key)
            examples['image'] = images
            examples['label'] = keys
            return examples
        dataset.set_transform(custom_trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)

        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, data_loader, find_thres=True)

