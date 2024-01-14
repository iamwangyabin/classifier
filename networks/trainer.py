import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import timm

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class Trainer(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.isTrain = opt.isTrain



        if self.isTrain and not opt.continue_train:
            if opt.arch == 'res50':
                self.model = timm.create_model('resnet50.tv_in1k', pretrained=True, num_classes=1)
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            elif opt.arch == 'vit':
                self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        else:

            if opt.arch == 'res50':
                self.model = resnet50(num_classes=1)
            elif opt.arch == 'vit':
                self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
            self.load_networks(opt.epoch)




    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y % 2
        x = x.to(self.dtype)
        # import pdb;pdb.set_trace()
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.to(self.dtype)
        logits = self.net(x)
        preds = torch.argmax(logits, dim=1) % 2
        test_loss = self.criterion(logits, y % 2)
        self.validation_step_outputs_preds.append(preds)
        self.validation_step_outputs_gts.append(y)
        self.log('val_loss', test_loss, on_step=True, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0)
        all_gts = torch.cat(self.validation_step_outputs_gts, 0)
        acc = self.accuracy(all_preds % 2, all_gts % 2)
        self.log('val_acc', acc, logger=True)

        for i, sub_task in enumerate(self.sub_tasks):
            mask = (all_gts >= i * 2) & (all_gts <= 1 + i * 2)
            idxes = torch.where(mask)[0]
            if len(idxes) == 0:
                continue
            acc = self.accuracy(all_preds[idxes] % 2, all_gts[idxes] % 2)
            self.log(f'val_acc_{sub_task}', acc, logger=True)
            # import pdb;pdb.set_trace()

        self.validation_step_outputs_preds.clear()  # free memory
        self.validation_step_outputs_gts.clear()  # free memory

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.SGD(optparams, momentum=0.9, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def init_weights(net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)


class Trainer(BaseModel):

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)




    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

