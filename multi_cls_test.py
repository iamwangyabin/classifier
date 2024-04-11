# 这段代码就是测试一下 CLIP在LSUN做多分类的精度
import os
from PIL import Image
import io
import csv
import argparse
import hydra
import pickle
from tqdm import tqdm
from utils.util import load_config_with_cli, archive_files
from networks.SPrompts.zsclip import ZeroshotCLIP_PE, ZeroshotCLIP

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


if __name__ == '__main__':
    # python multi_cls_test.py --cfg cfgs/train_mc_coop_progan.yaml
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop((224,224)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    order_classes = ["airplane", "bird", "bottle", "car", "chair", "diningtable", "horse", "person", "sheep", "train",
                     "bicycle", "boat", "bus", "cat", "cow", "dog", "motorbike", "pottedplant", "sofa", "tvmonitor"]
    class_to_idx = {class_name: i for i, class_name in enumerate(order_classes)}

    model = ZeroshotCLIP_PE(conf, order_classes, "cuda:0")
    model.cuda()
    model.eval()
    root_dir = "/home/jwang/ybwork/data/deepfake_benchmark/ForenSynths/val"

    dataset = LSUNDeepfakeBenchmarkDataset(root_dir, class_to_idx, selected_labels=["1_fake"], transform=transform)


    data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.dataset.train.batch_size,
                                              num_workers=conf.dataset.train.loader_workers)
    y_pred = []
    y_true = []
    with torch.no_grad():
        print("Length of dataset: %d" % (len(data_loader)))
        for img, label in tqdm(data_loader):
            in_tens = img.cuda()
            logits = model(in_tens)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(label.cpu().numpy())

    correct_predictions = sum(p == t for p, t in zip(y_pred, y_true))
    accuracy = correct_predictions / len(y_true)
    print(f'Accuracy: {accuracy:.4f}')

# vit/l
# Prompt ensemble: Accuracy: 0.9008 real    0.8413 fake
# No ensempl: Accuracy:                    0.8370 fake
# coop accuracy: real:0.922249972820282    fake: 0.8615000247    相比较人工设计的prompt都能提升一些，但是仅仅是拟合prompt

# vit/b
# Prompt ensemble: Accuracy: 0.8980 real    0.8452 fake
# pomp : 0.898500025 real 0.8447499871 fake









