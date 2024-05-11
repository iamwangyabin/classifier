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

from utils.util import load_config_with_cli
from networks.UniversalFakeDetect.clip_models import CLIPModel
from networks.resnet import resnet50
from networks.SPrompts.arprompts import ARPromptsCLIP

class ForenSynthsDataset(Dataset):
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
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx



class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for img_name in os.listdir(root_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', 'webp')):
                img_path = os.path.join(root_dir, img_name)
                self.samples.append((img_path, 0))

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=90, optimice=True)
        outputIoStream.seek(0)
        image = Image.open(outputIoStream)

        if self.transform:
            image = self.transform(image)
        return image, class_idx


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])



# model = CLIPModel('ViT-L/14')
# model = CLIPModel('RN50')
# model = resnet50(pretrained=True)

model = resnet50(num_classes=1)
state_dict = torch.load('networks/weights/blur_jpg_prob0.1.pth', map_location='cpu')
model.load_state_dict(state_dict['model'])


# parser = argparse.ArgumentParser(description='Testing')
# parser.add_argument('--cfg', type=str, default=None, required=True)
# args, cfg_args = parser.parse_known_args()
# conf = load_config_with_cli(args.cfg, args_list=cfg_args)
# conf = hydra.utils.instantiate(conf)
# def resume_lightning(model, conf):
#     if conf.resume:
#         state_dict = torch.load(conf.resume, map_location='cpu')['state_dict']
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if key.startswith('model.'):
#                 new_key = key[6:]  # remove `model.` from key
#                 new_state_dict[new_key] = value
#             else:
#                 new_state_dict[key] = value
#         model.load_state_dict(new_state_dict)
# model = ARPromptsCLIP(conf)
# resume_lightning(model, conf)
#
model.cuda()
model.eval()



# dataset = ForenSynthsDataset(root_dir='../data/DFBenchmark/ForenSynths/val', transform=transform, selected_realfake='0_real')
# dataset = ForenSynthsDataset(root_dir='../data/DFBenchmark/ForenSynths/val', transform=transform, selected_realfake='1_fake')

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/lsun_bedroom/real', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/lsun_bedroom/ldm', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/lsun_bedroom/midjourney', transform=transform)

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/celebahq/dalle2', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/celebahq/real', transform=transform)

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/ForenSynths/test/deepfake/0_real', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/ForenSynths/test/deepfake/1_fake', transform=transform)

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/AIGCDetect/stable_diffusion_v_1_5/1_fake', transform=transform)
dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/AIGCDetect/stable_diffusion_v_1_5/0_real', transform=transform)




data_loader = DataLoader(dataset, batch_size=512, shuffle=False)

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
all_features['features'] = [subset_features, y_true]


# with open('our881_DiffusionForensics_lsun_bedroom_mj_features.pkl', 'wb') as file:
#     pickle.dump(all_features, file)

with open('cnndet_AIGCDetect_sd15_real_features.pkl', 'wb') as file:
    pickle.dump(all_features, file)