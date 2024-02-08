import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.utils.data


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
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, num_real, num_fake
