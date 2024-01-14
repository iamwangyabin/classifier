import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image


from utils.validate import validate, get_val_opt
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt(opt)
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)






    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training dataloader steps = %d' % dataset_size)

    model = Trainer(opt)
    model.train()

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                # train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))




