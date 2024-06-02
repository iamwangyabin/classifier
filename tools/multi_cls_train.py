# 这段代码进行训练coop，在LSUN上
import os
from PIL import Image
import io
import csv
import argparse
import hydra
import pickle
import wandb
import datetime
from tqdm import tqdm


import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Accuracy


import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.util import load_config_with_cli, archive_files
from networks.SPrompts.zsclip import ZeroshotCLIP_PE, ZeroshotCLIP
from networks.SPrompts.coop import load_clip_to_cpu, CustomCLIP

class LSUNDeepfakeBenchmarkDataset(Dataset):

    def __init__(self, root_dir, class_to_idx, selected_labels=["0_real"], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.class_to_idx = class_to_idx

        self.classes = os.listdir(root_dir)
        for class_name in self.classes:
            for label in selected_labels:
                class_dir = os.path.join(root_dir, class_name, label)
                images_in_class = [(os.path.join(class_dir, img), self.class_to_idx[class_name]) for img in os.listdir(class_dir)]
                self.images.extend(images_in_class)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class CoOPPL(L.LightningModule):
    def __init__(self, cfg, classes):
        super().__init__()
        self.opt = cfg
        self.save_hyperparameters()
        print(f"Loading CLIP (backbone: {cfg.model.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.model = CustomCLIP(cfg, classes, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch

        tokenized_prompts = self.model.tokenized_prompts
        logit_scale = self.model.logit_scale.exp()

        prompts = self.model.prompt_learner()
        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        image_features = self.model.image_encoder(x)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        loss = self.criterion(logits, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        tokenized_prompts = self.model.tokenized_prompts
        logit_scale = self.model.logit_scale.exp()
        prompts = self.model.prompt_learner()
        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        image_features = self.model.image_encoder(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)

        pred = torch.argmax(logits, dim=1)
        correct_predictions = (pred==y).sum()
        accuracy = correct_predictions / len(y)

        self.log('val_acc', accuracy, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        tokenized_prompts = self.model.tokenized_prompts
        logit_scale = self.model.logit_scale.exp()
        prompts = self.model.prompt_learner()
        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        image_features = self.model.image_encoder(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        pred = torch.argmax(logits, dim=1)
        correct_predictions = (pred==y).sum()
        accuracy = correct_predictions / len(y)

        self.log('accuracy', accuracy, on_epoch=True)

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    # python multi_cls_train.py --cfg cfgs/train_mc_coop_progan.yaml
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)
    wandb.login(key = 'a4d3a740e939973b02ac59fbd8ed0d6a151df34b')

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    order_classes = ["airplane", "bird", "bottle", "car", "chair", "diningtable", "horse", "person", "sheep", "train",
                     "bicycle", "boat", "bus", "cat", "cow", "dog", "motorbike", "pottedplant", "sofa", "tvmonitor"]
    class_to_idx = {class_name: i for i, class_name in enumerate(order_classes)}

    train_dataset = LSUNDeepfakeBenchmarkDataset(conf.dataset.train.dataroot, class_to_idx, ["0_real"], transform=train_transforms)
    val_dataset = LSUNDeepfakeBenchmarkDataset(conf.dataset.val.dataroot, class_to_idx, ["1_fake"], transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=conf.dataset.train.batch_size, shuffle=True, num_workers=conf.dataset.train.loader_workers)
    val_loader = DataLoader(val_dataset, batch_size=conf.dataset.val.batch_size, shuffle=False, num_workers=conf.dataset.val.loader_workers)

    today_str = conf.name +"_"+ datetime.datetime.now().now().strftime('%Y%m%d_%H_%M_%S')

    wandb_logger = WandbLogger(name=today_str, project='DeepfakeDetection',
                               job_type='train', group=conf.name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('logs', today_str),
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    model = CoOPPL(conf, order_classes)
    trainer = L.Trainer(logger=wandb_logger, max_epochs=conf.train.train_epochs, accelerator="gpu", devices=conf.train.gpu_ids,
                        callbacks=[checkpoint_callback],
                        val_check_interval=0.5,
                        precision="16")

    weight = torch.load('./pomp.tar')['state_dict']['ctx']
    import pdb;pdb.set_trace()
    model.model.prompt_learner.ctx.data = weight
    trainer.test(model, dataloaders=val_loader)


    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #
    # trainer.save_checkpoint(os.path.join('logs', today_str, "last.ckpt"))

    # model = CoOPPL(conf, order_classes)
    # trainer = L.Trainer(accelerator="gpu", devices=conf.train.gpu_ids,precision="16")
    #
    # # trainer.fit(model, ckpt_path="/home/jwang/ybwork/classifier/logs/multiclass-ViTL—Progan_20240404_20_24_33/last.ckpt", train_dataloaders=train_loader)
    #





