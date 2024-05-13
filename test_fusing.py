# code for testing fusing only

import argparse
import csv
import io
import json
import os
import copy
import pickle
import sys
import warnings
import cv2
import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from random import choice, random
from scipy.fftpack import dct
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score
from copy import deepcopy
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


from utils.util import load_config_with_cli, archive_files
from utils.network_factory import get_model

class BinaryJsonDatasets(Dataset):
    def __init__(self, opt, data_root, subset='all', split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.CropSize = opt.CropSize
        self.qf = opt.qf

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        if self.qf:
            outputIoStream = io.BytesIO()
            image.save(outputIoStream, "JPEG", quality=self.qf, optimice=True)
            outputIoStream.seek(0)
            image = Image.open(outputIoStream)

        height, width = image.height, image.width

        input_img = copy.deepcopy(image)
        input_img = transforms.ToTensor()(input_img)
        input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

        image = transforms.Resize(self.CropSize)(image)
        image = transforms.CenterCrop(self.CropSize)(image)
        cropped_img = transforms.ToTensor()(image)
        cropped_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(cropped_img)

        scale = torch.tensor([height, width])

        label = self.labels[idx]
        return input_img, cropped_img, scale, label


def patch_collate_test(batch):
    input_img=[item[0] for item in batch]
    cropped_img=torch.stack([item[1] for item in batch], dim=0)
    scale=torch.stack([item[2] for item in batch], dim=0)
    target=torch.tensor([item[3] for item in batch])
    return [input_img, cropped_img, scale, target]


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


def validate_PSM(model, data_loader):
    y_true, y_pred, y_logits = [], [], []
    i = 0
    with torch.no_grad():
        for data in data_loader:
            i += 1
            print("batch number {}/{}".format(i, len(data_loader)), end='\r')
            input_img = data[0]  # [batch_size, 3, height, width]
            cropped_img = data[1].cuda()  # [batch_size, 3, 224, 224]
            scale = data[2].cuda()  # [batch_size, 1, 2]
            label = data[3].cuda()  # [batch_size, 1]
            logits = model(input_img, cropped_img, scale)
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
            y_logits.extend(logits.flatten().tolist())

    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0
    try:
        f1 = f1_score(y_true, y_pred>0.5)
    except:
        f1 = 0

    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = {
        'ap': ap,
        'auc': auc,
        'f1': f1,
        'r_acc0': r_acc0,
        'f_acc0': f_acc0,
        'acc0': acc0,
        'num_real': num_real,
        'num_fake': num_fake,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_logits': y_logits
    }
    return result_dict

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = get_model(conf)
    model.cuda()
    model.eval()
    all_results = []
    save_raw_results = {}
    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                        collate_fn=patch_collate_test, num_workers=conf.datasets.loader_workers)

            result = validate_PSM(model, data_loader)

            ap = result['ap']
            auc = result['auc']
            f1 = result['f1']
            r_acc0 = result['r_acc0']
            f_acc0 = result['f_acc0']
            acc0 = result['acc0']
            num_real = result['num_real']
            num_fake = result['num_fake']

            print(f"{set_name} {subset}")
            print(
                f"AP: {ap:.4f},\tF1: {f1:.4f},\tAUC: {auc:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([set_name, subset, ap, auc, f1, r_acc0, f_acc0, acc0, num_real, num_fake])
            save_raw_results[f"{set_name} {subset}"] = result

    columns = ['dataset', 'sub_set', 'ap', 'auc', 'f1', 'r_acc0', 'f_acc0', 'acc0', 'num_real', 'num_fake']
    with open(conf.test_name + '_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
    with open(conf.test_name + '.pkl', 'wb') as file:
        pickle.dump(save_raw_results, file)
