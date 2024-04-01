import io
import os
import csv
import argparse
import hydra
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from utils.util import load_config_with_cli, archive_files
from utils.validate import validate
from data.json_datasets import BinaryJsonDatasets
from data.binary_datasets import BinaryMultiDatasets
from utils.network_factory import get_model
from networks.UniversalFakeDetect.clip_models import CLIPModel

def multi_class_progan_trainset():
    class DeepFakeBenchmarkDataset(Dataset):
        def __init__(self, root_dir, transform=None, selected_realfake='1_fake'):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.samples = []
            for class_name in self.classes:
                class_idx = self.class_to_idx[class_name]
                real_dir_path = os.path.join(self.root_dir, class_name, selected_realfake)
                for img_name in os.listdir(real_dir_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', 'webp')):
                        img_path = os.path.join(real_dir_path, img_name)
                        self.samples.append((img_path, class_idx))
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img_path, class_idx = self.samples[idx]
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, class_idx

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = DeepFakeBenchmarkDataset(root_dir='../data/deepfake_benchmark/ForenSynths/train', transform=transform)

    data_loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model = CLIPModel('ViT-L/14')
    model.cuda()
    model.eval()

    all_features = {}

    subset_features = []
    y_true = []
    with torch.no_grad():
        print("Length of dataset: %d" % (len(data_loader)))
        for img, label in tqdm(data_loader):
            in_tens = img.cuda()
            features = model(in_tens, return_feature=True)[1]
            subset_features.extend(features.tolist())
            y_true.extend(label.tolist())


    subset_features = np.array(subset_features)
    y_true = np.array(y_true)
    all_features[f"ProGAN_train_fake"] = [subset_features, y_true]

    with open('clip_progan_train_multicls_fake_features.pkl', 'wb') as file:
        pickle.dump(all_features, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = get_model(conf)
    model.cuda()
    model.eval()

    all_features = {}
    dataset = BinaryMultiDatasets(conf.dataset.train, split='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                              num_workers=conf.datasets.loader_workers)
    subset_features = []
    y_true = []
    with torch.no_grad():
        print("Length of dataset: %d" % (len(data_loader)))
        for img, label in tqdm(data_loader):
            in_tens = img.cuda()
            features = model(in_tens, return_feature=True)[1]
            subset_features.extend(features.tolist())
            y_true.extend(label.tolist())
    subset_features = np.array(subset_features)
    y_true = np.array(y_true)
    all_features[f"ProGAN_train"] = [subset_features, y_true]

    with open(conf.test_name + 'progan_train_features.pkl', 'wb') as file:
        pickle.dump(all_features, file)


    all_features = {}
    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            # dataset = BinaryMultiDatasets(conf.dataset.train, split='train')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                                      num_workers=conf.datasets.loader_workers)
            subset_features = []
            y_true = []
            with torch.no_grad():
                print("Length of dataset: %d" % (len(data_loader)))
                for img, label in tqdm(data_loader):
                    in_tens = img.cuda()
                    features = model(in_tens, return_feature=True)[1]
                    subset_features.extend(features.tolist())
                    y_true.extend(label.tolist())
            subset_features = np.array(subset_features)
            y_true = np.array(y_true)
            all_features[f"{set_name} {subset}"] = [subset_features, y_true]

    with open(conf.test_name + 'test_features.pkl', 'wb') as file:
        pickle.dump(all_features, file)
