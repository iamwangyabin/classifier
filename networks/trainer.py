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


class Trainer_multicls(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)
        self.validation_step_outputs_preds.append(F.softmax(logits, 1)[:,1])
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(torch.float32).flatten().cpu().numpy()
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




mapping = {0: 0, 1: 20, 2: 1, 3: 21, 4: 2, 5: 22, 6: 3, 7: 23, 8: 4, 9: 24, 10: 5, 11: 25, 12: 6, 13: 26, 14: 7, 15: 27, 16: 8, 17: 28, 18: 9, 19: 29, 20: 10, 21: 30, 22: 11, 23: 31, 24: 12, 25: 32, 26: 13, 27: 33, 28: 14, 29: 34, 30: 15, 31: 35, 32: 16, 33: 36, 34: 17, 35: 37, 36: 18, 37: 38, 38: 19, 39: 39}

class Trainer_arpmulticls(L.LightningModule):
    # 这个实现相比较multicls是将每个类的real fake都堪称独立的类别
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.celoss = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch
        logits, b_logits = self.model(x, return_binary=True)
        loss = 0
        # 首先语义对齐：将y归为20个类，然后logits向这20类输出监督，无关real/fake  实现为对logits维度进行切分，每20个一组进行cross entropy
        cls_y = y//2 #0~40类 -> 0~19类 如果prompt是2，如果是1，那么就是
        logits_groups = torch.chunk(logits, 2*self.opt.model.PROMPT_NUM_TEXT, dim=1)
        for i, logits_group in enumerate(logits_groups):
            loss += F.cross_entropy(logits_group, cls_y)

        # 其次次级语义对齐：fake同fake的分类对齐 real同real的对齐 也是ce损失
        # 实现为通过chunk和mask进行，将属于real的样本mask掉，
        # for i, logits_group in enumerate(logits_groups):
        #     mask = i//2 == y % 2
        #     loss += 0.5*F.cross_entropy(logits_group[mask], cls_y[mask])

        # import pdb;pdb.set_trace()
        # 在每个子空间做deepfake detection，也就是以类为单位进行deepfake detection
        # 0表示real，1表示fake，这是个deepfake detection任务，因此
        # 主要问题是我们的y是0，1，2，3，4，5...38，39这样的，每个类都有两个label，分别表示real和fake，这样进行了20个类，表现为40个label，也就是每个类都有real和fake
        # 然而logits是0-19个real，之后是20-39个fake，（prompt=1时候），也就是先输出每个类的real再输出每个类的fake
        # 我们需要变换y，让它和logits对齐
        new_y =  torch.tensor([mapping[label.item()] for label in y], dtype=torch.long, device=y.device)
        loss += F.cross_entropy(logits, new_y)

        # 其次任务对齐： 把所有20个类统一成real/fake，然后所有logits也统一成real/fake（prompts加起来）
        # int(y % 2)
        # general real prompt的logits
        # general fake prompt的logits
        loss += F.cross_entropy(b_logits, y % 2)
        # loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model.forward_binary(x)

        # import pdb;pdb.set_trace()

        self.validation_step_outputs_preds.append(F.softmax(logits, 1)[:,1])
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(torch.float32).flatten().cpu().numpy()
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





