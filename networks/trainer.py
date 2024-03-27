import os
import functools
import timm
import hydra
import numpy as np

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler

from utils.util import validate
from utils.network_factory import get_model


class Trainer(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []

        self.criterion = nn.BCEWithLogitsLoss()
        # if opt.arch == 'sp_l':
        #     self.criterion = nn.CrossEntropyLoss()
        #
        #     enabled = set()
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             enabled.add(name)
        #     print(f"Parameters to be updated: {enabled}")


    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        if self.opt.arch == 'sp_l':
            loss = F.cross_entropy(logits, y)
        else:
            loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        keyargwords={'inference': True}
        logits = self.model(x, **keyargwords)
        # if self.opt.arch == 'sp_l':
        #     test_loss = F.cross_entropy(logits, y%2)
        #     self.validation_step_outputs_preds.append(torch.softmax(logits,1)[:,0])
        # else:
        # test_loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)
        # self.log('val_loss', test_loss, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        # if self.opt.arch == 'sp_l':
        #     all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(
        #         torch.float32).flatten().cpu().numpy()
        # else:
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(
                torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap = validate(all_gts % 2, all_preds)[:2]
        self.log('val_acc_epoch', acc, logger=True, sync_dist=True)
        for i, sub_task in enumerate(self.opt.dataset.val.subfolder_names):
            mask = (all_gts >= i * 2) & (all_gts <= 1 + i * 2)
            idxes = np.where(mask)[0]
            if len(idxes) == 0:
                continue
            acc, ap = validate(all_gts[idxes] % 2, all_preds[idxes])[:2]
            self.log(f'val_acc_{sub_task}', acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]
