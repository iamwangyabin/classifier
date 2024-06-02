import argparse
import os
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm

import timm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



def get_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', nargs='+', type=str, default='examples/realfakedir', help='input a dir or a image path')
    parser.add_argument('--model_path', type=str, default='weights.pth')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='model name from timm repo, for example: vit_base_patch16_224, resnet50')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of workers')
    parser.add_argument('-c', '--crop', type=int, default=None, help='by default, do not crop. specify crop size')
    parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
    opt = parser.parse_args()
    return opt


class FlexibleImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        """
        Args:
            paths (list): List of paths to images or directories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_paths = []
        for path in paths:
            if os.path.isdir(path):
                self.image_paths.extend([os.path.join(path, file) for file in os.listdir(path) if self.is_image_file(file)])
            elif os.path.isfile(path) and self.is_image_file(path):
                self.image_paths.append(path)
        self.transform = transform

    def is_image_file(self, filename):
        """Checks if a file is an image."""
        IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image



if __name__ == '__main__':
    opt = get_opt()

    # Load model
    model = timm.create_model(opt.model_name, pretrained=True, num_classes=1)
    if (opt.model_path is not None):
        state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()
    if (not opt.use_cpu):
        model.cuda()

    # Data Transform
    trans_init = []
    if (opt.crop is not None):
        trans_init = [transforms.CenterCrop(opt.crop), ]
        print('Cropping to [%i]' % opt.crop)
    else:
        print('Not cropping')
    trans = transforms.Compose(trans_init + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FlexibleImageDataset(paths=opt.input, transform=trans)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    y_pred = []
    with torch.no_grad():
        for data in tqdm(loader):
            y_pred.extend(model(data).sigmoid().flatten().tolist())

    print('probability of being synthetic: {:.2f}%'.format(prob * 100))
