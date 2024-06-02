import os
import warnings
from PIL import ImageFile, Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data.augmentations import data_augment

warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class BinaryMultiDatasets(Dataset):
    def __init__(self, opt, split='train'):
        self.dataroot = opt.dataroot
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.label_mapping = {'0_real': 0, '1_fake': 1}
        image_extensions = ('.jpg', '.jpeg', '.png')

        for id, subfolder in enumerate(opt.subfolder_names):
            if opt.multicalss_idx[id]:
                classes = os.listdir(os.path.join(self.dataroot, subfolder, split))
            else:
                classes = ['']
            for cls in classes:
                root = os.path.join(self.dataroot, subfolder, split, cls)
                for label in ['0_real', '1_fake']:
                    label_dir = os.path.join(root, label)
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        if img_path.lower().endswith(image_extensions):
                            self.image_pathes.append(img_path)
                            self.labels.append(self.label_mapping[label]+id*2)

        if split == 'train':
            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.RandomResizedCrop(opt.cropSize),
                transforms.RandomHorizontalFlip() if opt.random_flip else transforms.Lambda(lambda img: img),
                transforms.Lambda(lambda img: data_augment(img, opt)) if opt.augment else transforms.Lambda(lambda img: img),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]

        else:
            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.CenterCrop(opt.cropSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]

        self.transform_chain = transforms.Compose(trsf)


    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform_chain(image)
        return image, label


class BinarySingleDataset(Dataset):
    """
    Dataset for binary classification, merge multiple datasets into one.
    Label: 0 for real, 1 for fake, no increase for multiclass.
    """
    def __init__(self, opt, task_names, multicalss_idx, split='train'):
        self.dataroot = opt.dataroot
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.label_mapping = {'0_real': 0, '1_fake': 1}
        image_extensions = ('.jpg', '.jpeg', '.png')

        for id, subfolder in enumerate(task_names):
            if multicalss_idx[id]:
                classes = os.listdir(os.path.join(self.dataroot, subfolder, split))
            else:
                classes = ['']
            for cls in classes:
                root = os.path.join(self.dataroot, subfolder, split, cls)
                for label in ['0_real', '1_fake']:
                    label_dir = os.path.join(root, label)
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        if img_path.lower().endswith(image_extensions):
                            self.image_pathes.append(img_path)
                            self.labels.append(self.label_mapping[label]+id*2)

        if split == 'train':
            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.RandomResizedCrop(opt.cropSize),
                transforms.RandomHorizontalFlip() if opt.random_flip else transforms.Lambda(lambda img: img),
                transforms.Lambda(lambda img: data_augment(img, opt.augment)) if opt.augment else transforms.Lambda(lambda img: img),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]

        else:
            trsf = [
                transforms.Resize(opt.loadSize),
                transforms.CenterCrop(opt.cropSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]

        self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform_chain(image)
        return image, label
