import io
import csv
import argparse
from PIL import Image
import hydra
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from copy import deepcopy
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torch.utils.data

# from datasets import load_dataset

from networks.UniversalFakeDetect.clip_models import CLIPModel, CLIPModel_inc
from utils.util import load_config_with_cli, archive_files
from data.json_datasets import BinaryJsonDatasets


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
            # img, label = sample['image'], sample['label']
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    print("Model loaded..")

    model = CLIPModel('ViT-L/14')
    state_dict = torch.load('networks/UniversalFakeDetect/fc_weights.pth', map_location='cpu')
    model.fc.load_state_dict(state_dict)

    # model = CLIPModel_inc('ViT-L/14')
    # state_dict = torch.load('/home/jwang/ybwork/classifier/logs/20240129_20_21_23/epoch=41-val_acc_epoch=0.84.ckpt', map_location='cpu')
    # model.fc.weight.data = state_dict['state_dict']['model.fc.weight']
    # model.fc.bias.data = state_dict['state_dict']['model.fc.bias']

    model.eval()
    model.cuda()

    all_results = []

    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                                      num_workers=conf.datasets.loader_workers)
            ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, data_loader)
            print(f"{set_name} {subset}")
            print(f"AP: {ap:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([set_name, subset, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres])

    columns = ['dataset', 'sub_set', 'ap', 'r_acc0', 'f_acc0', 'acc0', 'r_acc1', 'f_acc1', 'acc1', 'best_thres']
    with open('model_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
