# Swin transformer as detection

from datasets import load_dataset
import torch
from deepfakes_dataset import DeepFakesDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import yaml
import argparse
import math
from statistics import mean
from progress.bar import ChargingBar
from albumentations import Cutout, CoarseDropout, RandomCrop, RandomGamma, MedianBlur, ISONoise, MultiplicativeNoise, \
    ToSepia, RandomShadow, MultiplicativeNoise, RandomSunFlare, GlassBlur, RandomBrightness, MotionBlur, RandomRain, \
    RGBShift, RandomFog, RandomContrast, Downscale, InvertImg, RandomContrast, ColorJitter, Compose, \
    RandomBrightnessContrast, CLAHE, ISONoise, JpegCompression, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, \
    ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate, Normalize, Resize
from timm.scheduler.cosine_lr import CosineLRScheduler
import cv2
from multiprocessing import Manager
from multiprocessing.pool import Pool
from functools import partial
import numpy as np
from cross_efficient_vit import CrossEfficientViT
import math
import random
import os
from utils import check_correct, unix_time_millis, custom_round
from datetime import datetime, timedelta
from torchvision.models import resnet50, ResNet50_Weights
import glob
import pandas as pd
import collections
from sklearn.metrics import f1_score, roc_curve, auc


import torch
import timm

model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', in_chans=3, pretrained=True)
model.head.fc = torch.nn.Linear(1024, 1)
model.load_state_dict(torch.load(opt.model1_weights))


test_dataset = DeepFakesDataset(test_paths, test_labels, config['model']['image-size'], methods=test_methods,
                                mode='test')

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['test']['bs'], shuffle=False, sampler=None,
                                      batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                      pin_memory=False, drop_last=False, timeout=0,
                                      worker_init_fn=None, prefetch_factor=2,
                                      persistent_workers=False)


    bar = ChargingBar('PREDICT', max=(len(test_dl)))
    preds = []
    names = []
    test_counter = 0
    test_correct = 0
    test_positive = 0
    test_negative = 0
    correct_test_labels = []
    for index, (images, images_dct, image_path, labels, methods) in enumerate(test_dl):
        with torch.no_grad():

            labels = labels.unsqueeze(1)
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(device)

            y_pred = model(images)
            y_pred = y_pred.cpu()
            preds.extend(torch.sigmoid(torch.tensor(y_pred)))
            correct_test_labels.extend(labels)
            names.append(os.path.basename(image_path[0]))

            corrects, positive_class, negative_class = check_correct(y_pred, labels, opt.ensemble, opt.threshold)



            test_correct += corrects
            test_positive += positive_class
            test_counter += 1
            test_negative += negative_class

            bar.next()
