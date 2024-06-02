import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score

from copy import deepcopy
from tqdm import tqdm

import torch
import torch.utils.data
from torch.nn import functional as F


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
        y_true, y_pred, y_logits = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()
            logits = model(in_tens)
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
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



def validate_multicls(model, loader):
    # 这就是个两个分类头的binary 分类器该如何实现val
    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()
            logits = model(in_tens)
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(F.softmax(logits, 1)[:,1].flatten().tolist())
            y_true.extend(label.flatten().tolist())
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


def validate_arp(model, loader, specific_cls):

    if specific_cls:
        from networks.SPrompts.arprompts import load_clip_to_cpu
        clip_model = load_clip_to_cpu(model.cfg)
        token_embedding = clip_model.token_embedding


    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()
            if specific_cls:
                logits = model.forward_binary_classnames(in_tens, specific_cls, token_embedding)
            else:
                logits = model.forward_binary(in_tens)
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(F.softmax(logits, 1)[:,1].flatten().tolist())
            y_true.extend(label.flatten().tolist())
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
