# code for testing LGrad only

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
import torch.nn.functional as F
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
from networks.LGrad import build_model

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
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
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


def normlize_np(img):
    img -= img.min()
    if img.max() != 0: img /= img.max()
    return img * 255.


def validate(model_dis, model_gen, data_loader):
    y_true, y_pred, y_logits = [], [], []
    i = 0
    for data in data_loader:
        i += 1
        print("batch number {}/{}".format(i, len(data_loader)), end='\r')
        input_img = data[0].cuda().to(torch.float32)  # [batch_size, 3, height, width]
        label = data[1].cuda()  # [batch_size, 1]
        # print(data[2])
        input_img.requires_grad = True
        pre = model_gen(input_img)
        model_gen.zero_grad()
        grads = torch.autograd.grad(pre.sum(), input_img, create_graph=True, retain_graph=True, allow_unused=False)[0]

        b_min = torch.min(grads.view(grads.size(0), -1), dim=1)[0]
        grads = grads - b_min.view(-1, 1, 1, 1)
        b_max = torch.max(grads.view(grads.size(0), -1), dim=1)[0]
        grads = grads/b_max.view(-1, 1, 1, 1)
        grads = grads*255.

        grads = F.interpolate(grads, size=(224, 224), mode='bilinear', align_corners=False)

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        mean = torch.tensor(MEAN).view(1, 3, 1, 1).cuda() # Reshape to [1, 3, 1, 1] for broadcasting
        std = torch.tensor(STD).view(1, 3, 1, 1).cuda() # Reshape to [1, 3, 1, 1] for broadcasting
        # import pdb;pdb.set_trace()
        grads = torch.clamp(grads, 0, 255) / 255.0
        normalized_grads = (grads - mean) / std
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            logits = model_dis(normalized_grads)
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


    gen_model = build_model(gan_type='stylegan', module='discriminator', resolution=256, label_size=0, image_channels=3)
    gen_model.load_state_dict(torch.load(conf.LGrad_modelpath), strict=True)
    gen_model.cuda()
    gen_model.eval()

    # for name, param in gen_model.named_parameters():
    #     param.requires_grad = True

    all_results = []
    save_raw_results = {}
    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                    num_workers=conf.datasets.loader_workers)

            result = validate(model, gen_model, data_loader)

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


