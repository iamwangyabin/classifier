import os
import hydra
import argparse
import wandb
import datetime

import torch
import torch.nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.util import load_config_with_cli, archive_files
from data.binary_datasets import BinaryMultiDatasets
from networks.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)
    wandb.login(key = 'a4d3a740e939973b02ac59fbd8ed0d6a151df34b')

    train_dataset = BinaryMultiDatasets(conf.dataset.train, split='train')
    train_loader = DataLoader(train_dataset, batch_size=conf.dataset.train.batch_size, shuffle=True,
                              num_workers=conf.dataset.train.loader_workers)
    val_dataset = BinaryMultiDatasets(conf.dataset.val, split='val')
    val_loader = DataLoader(val_dataset, batch_size=conf.dataset.val.batch_size, shuffle=False,
                            num_workers=conf.dataset.val.loader_workers)

    today_str = conf.name +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')

    wandb_logger = WandbLogger(name=today_str, project='DeepfakeDetection',
                               job_type='train', group=conf.name)

    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git'])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_epoch',
        dirpath=os.path.join('logs', today_str),
        filename='{epoch:02d}-{val_acc_epoch:.2f}',
        save_top_k=1,
        mode='max',
    )

    model = Trainer(opt=conf)
    trainer = L.Trainer(logger=wandb_logger, max_epochs=conf.train.train_epochs, accelerator="gpu", devices=conf.train.gpu_ids,
                        callbacks=[checkpoint_callback],
                        val_check_interval=0.1,
                        # check_val_every_n_epoch=conf.train.check_val_every_n_epoch,
                        precision="16")


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(os.path.join('logs', today_str, "last.ckpt"))



