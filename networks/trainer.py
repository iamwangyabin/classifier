import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import timm

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            if opt.arch == 'res50':
                self.model = resnet50(pretrained=True)
                self.model.fc = nn.Linear(2048, 1)
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            elif opt.arch == 'vit':
                # from networks.vit import vit_base_patch16_224
                self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
                # self.model = vit_base_patch16_224(pretrained=True)
                # self.model.fc = nn.Linear(768, 1)
                # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            else:
                exit()

        if not self.isTrain or opt.continue_train:
            if opt.arch == 'res50':
                self.model = resnet50(num_classes=1)
            elif opt.arch == 'vit':
                self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
            else:
                exit()

        if self.isTrain:
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

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])


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

