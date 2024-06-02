import os
import json
import warnings
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import ImageFile, Image


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')



class BinaryJsonDatasets(Dataset):
    def __init__(self, opt, data_root, subset='all', split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            if os.path.exists(img_full_path):
                self.image_pathes.append(img_full_path)
                self.labels.append(label)
            else:
                print('pass, not exit {}'.format(img_full_path) )

        self.transform_chain = transforms.Compose(opt.trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform_chain(image)
        return image, label

