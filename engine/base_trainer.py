import os
import functools
import timm
import hydra
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
import lightning as L

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


    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)
        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)


    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(
                torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        self.log('val_acc_epoch', acc, logger=True, sync_dist=True)
        self.log('val_ap_epoch', ap, logger=True, sync_dist=True)
        self.log('val_racc_epoch', r_acc, logger=True, sync_dist=True)
        self.log('val_facc_epoch', f_acc, logger=True, sync_dist=True)
        # for i, sub_task in enumerate(self.opt.datasets.val.sub_sets):
        #     mask = (all_gts >= i * 2) & (all_gts <= 1 + i * 2)
        #     idxes = np.where(mask)[0]
        #     if len(idxes) == 0:
        #         continue
        #     acc, ap = validate(all_gts[idxes] % 2, all_preds[idxes])[:2]
        #     self.log(f'val_acc_{sub_task}', acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()


    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]
