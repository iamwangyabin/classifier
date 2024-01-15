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

class Trainer(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = timm.create_model(opt.arch, pretrained=True, num_classes=1)
        torch.nn.init.xavier_uniform_(self.model.head.weight.data)
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
        self.log('val_loss', test_loss, on_step=True, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).sigmoid().flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).cpu().numpy()
        acc, ap = validate(all_gts % 2, all_preds)[:2]
        self.log('val_acc', acc, logger=True)
        for i, sub_task in enumerate(self.opt.dataset.val.subfolder_names):
            mask = (all_gts >= i * 2) & (all_gts <= 1 + i * 2)
            idxes = np.where(mask)[0]
            if len(idxes) == 0:
                continue
            acc, ap = validate(all_gts[idxes] % 2, all_preds[idxes])[:2]
            self.log(f'val_acc_{sub_task}', acc, logger=True)
        self.validation_step_outputs_preds.clear()  # free memory
        self.validation_step_outputs_gts.clear()  # free memory

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]

    #
    # def init_weights(net, init_type='normal', gain=0.02):
    #     def init_func(m):
    #         classname = m.__class__.__name__
    #         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, gain)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=gain)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=gain)
    #             else:
    #                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)
    #         elif classname.find('BatchNorm2d') != -1:
    #             init.normal_(m.weight.data, 1.0, gain)
    #             init.constant_(m.bias.data, 0.0)
    #
    #     print('initialize network with %s' % init_type)
    #     net.apply(init_func)

