import os
import functools
import timm
import hydra
import numpy as np

import lightning as L
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from utils.util import validate

def get_network(arch_name):
    if arch_name == 'clip':
        from networks.UniversalFakeDetect.clip_models import CLIPModel
        model = CLIPModel('ViT-L/14')
    elif arch_name == 'clip_res':
        from networks.UniversalFakeDetect.clip_models import CLIPModel
        model = CLIPModel('RN50x64')
    elif arch_name == 'clip_vit336':
        from networks.UniversalFakeDetect.clip_models import CLIPModel
        model = CLIPModel('ViT-L/14@336px')
    else:
        model = timm.create_model(arch_name, pretrained=True, num_classes=1)
        torch.nn.init.xavier_uniform_(model.head.weight.data)
    return model


class Trainer(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = get_network(opt.arch)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)
        test_loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)
        self.log('val_loss', test_loss, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap = validate(all_gts % 2, all_preds)[:2]
        self.log('val_acc_epoch', acc, logger=True)
        for i, sub_task in enumerate(self.opt.dataset.val.subfolder_names):
            mask = (all_gts >= i * 2) & (all_gts <= 1 + i * 2)
            idxes = np.where(mask)[0]
            if len(idxes) == 0:
                continue
            acc, ap = validate(all_gts[idxes] % 2, all_preds[idxes])[:2]
            self.log(f'val_acc_{sub_task}', acc, logger=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]
