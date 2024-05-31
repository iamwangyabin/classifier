# code for testing LNP only

import argparse
import csv
import io
import json
import os
import pickle
import sys
import warnings
import hydra
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageFile
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, f1_score
from random import randint

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from omegaconf import ListConfig

from utils.util import load_config_with_cli
from utils.network_factory import get_model
from networks.LNP.denoising_rgb import DenoiseNet

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


class BinaryJsonDatasets(Dataset):
    def __init__(self, opt, data_root, subset='all', split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.qf = opt.qf

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        if self.qf:
            if isinstance(self.qf, ListConfig) and len(self.qf) == 2:
                outputIoStream = io.BytesIO()
                quality_factor = randint(int(self.qf[0]), int(self.qf[1]))
                image.save(outputIoStream, "JPEG", quality=quality_factor, optimize=True)
                outputIoStream.seek(0)
                image = Image.open(outputIoStream)
            else:
                outputIoStream = io.BytesIO()
                image.save(outputIoStream, "JPEG", quality=self.qf, optimice=True)
                outputIoStream.seek(0)
                image = Image.open(outputIoStream)

        image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_path


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def validate(model_dis, data_loader):
    y_true, y_pred, y_logits = [], [], []
    i = 0
    for data in data_loader:
        i += 1
        print("batch number {}/{}".format(i, len(data_loader)), end='\r')
        input_img = data[0].cuda().to(torch.float32)
        label = data[1].cuda()

        rgb_restored = model_restoration(input_img)
        rgb_restored = torch.round(torch.clamp(rgb_restored, 0, 1)*255.)*255.
        rgb_restored = torch.clamp(rgb_restored, 0, 255) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        rgb_restored = (rgb_restored - mean) / std

        with torch.no_grad():
            logits = model_dis(rgb_restored)

        # import pdb;pdb.set_trace()

        y_logits.extend(logits.flatten().tolist())
        y_pred.extend(logits.sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())

    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    ap = average_precision_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0
    try:
        f1 = f1_score(y_true, y_pred>0.5)
    except:
        f1 = 0

    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

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

    model_restoration = DenoiseNet()
    load_checkpoint(model_restoration, conf.LNP_modelpath)
    model_restoration.cuda()
    model_restoration.eval()


    all_results = []
    save_raw_results = {}
    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                    num_workers=conf.datasets.loader_workers)

            result = validate(model, data_loader)

            ap = result['ap']
            auc = result['auc']
            f1 = result['f1']
            r_acc0 = result['r_acc0']
            f_acc0 = result['f_acc0']
            acc0 = result['acc0']
            num_real = result['num_real']
            num_fake = result['num_fake']

            print(f"{set_name} {subset}")
            print(f"AP: {ap:.4f},\tF1: {f1:.4f},\tAUC: {auc:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([set_name, subset, ap, auc, f1, r_acc0, f_acc0, acc0, num_real, num_fake])
            save_raw_results[f"{set_name} {subset}"] = result

    columns = ['dataset', 'sub_set', 'ap', 'auc', 'f1', 'r_acc0', 'f_acc0', 'acc0', 'num_real', 'num_fake']
    with open(conf.test_name+'_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
    with open(conf.test_name + '.pkl', 'wb') as file:
        pickle.dump(save_raw_results, file)


