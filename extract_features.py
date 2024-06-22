import io
import os
import csv
import json
import argparse
import hydra
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from utils.util import load_config_with_cli
from networks.UniversalFakeDetect.clip_models import CLIPModel
from networks.resnet import resnet50
from networks.SPrompts.arprompts import ARPromptsCLIP
from networks.ClipBased.openclipnet import OpenClipLinear

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



class BinaryJsonDatasets(Dataset):
    def __init__(self, data_root, transform=None, subset='all', split='train'):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []
        self.transform = transform

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            self.image_pathes.append(img_full_path)
            self.labels.append(label)


    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label


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
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

def get_model(modelname):
    if modelname == 'clipvitl':
        model = CLIPModel('ViT-L/14')
    elif modelname == 'clipresnet50':
        model = CLIPModel('RN50')
    elif modelname == 'resnet50':
        model = resnet50(pretrained=True)
    elif modelname == 'cnndet':
        model = resnet50(num_classes=1)
        state_dict = torch.load('networks/weights/blur_jpg_prob0.1.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    elif modelname == 'poundnet':
        parser = argparse.ArgumentParser(description='Testing')
        parser.add_argument('--cfg', type=str, default=None, required=True)
        args, cfg_args = parser.parse_known_args()
        conf = load_config_with_cli(args.cfg, args_list=cfg_args)
        conf = hydra.utils.instantiate(conf)
        def resume_lightning(model, conf):
            if conf.resume:
                state_dict = torch.load(conf.resume, map_location='cpu')['state_dict']
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # remove `model.` from key
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                model.load_state_dict(new_state_dict)
        model = ARPromptsCLIP(conf)
        resume_lightning(model, conf)
    elif modelname == 'clipL14commonpool_next_to_last':
        model = OpenClipLinear(num_classes=1, pretrain='clipL14commonpool', normalize=False, next_to_last=True)
        model.bb[0].visual.output_tokens = True
    elif modelname == 'clipL14commonpool_last':
        model = OpenClipLinear(num_classes=1, pretrain='clipL14commonpool', normalize=False, next_to_last=False)
        model.bb[0].visual.output_tokens = True
    elif modelname == 'clipL14openai_next_to_last':
        model = OpenClipLinear(num_classes=1, pretrain='clipL14openai', normalize=False, next_to_last=True)
        model.bb[0].visual.output_tokens = True
    elif modelname == 'clipL14openai_last':
        model = OpenClipLinear(num_classes=1, pretrain='clipL14openai', normalize=False, next_to_last=False)
        model.bb[0].visual.output_tokens = True

    return model


for rf_tag in ['0_real', '1_fake']:
    # dataset = ForenSynthsDataset(root_dir='../data/DFBenchmark/ForenSynths/train', transform=transform, selected_realfake=rf_tag)
    # parquet_file = os.path.join('.', r'clipL14openai_next_to_last_progan_train_multicls_{}_features.parquet'.format(rf_tag))
    dataset = ForenSynthsDataset(root_dir='/scratch/yw26g23/datasets/deepfakebenchmark/ForenSynths/train', transform=transform, selected_realfake=rf_tag)
    parquet_file = os.path.join('/scratch/yw26g23', r'clipL14openai_next_to_last_progan_train_multicls_{}_features.parquet'.format(rf_tag))

    BATCH_ACCUMULATION = 128

    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=32)

    model = get_model('clipL14openai_next_to_last')
    model.cuda()
    model.to('cuda')
    model.eval()

    parquet_schema = pa.schema([
        ('label', pa.int32()),
        ('cls_tokens', pa.list_(pa.float32())),
        ('visual_tokens', pa.list_(pa.list_(pa.float32()))),
    ])

    parquet_writer = None
    accumulated_tables = []
    accumulated_data = {
        'label': [],
        'cls_tokens': [],
        'visual_tokens': []
    }

    with torch.no_grad():
        print("Length of dataset: %d" % len(data_loader))
        for batch_index, (img, label) in enumerate(tqdm(data_loader)):
            in_tens = img.cuda(non_blocking=True)
            cls_tokens, visual_tokens = model.forward_features(in_tens)

            # Accumulate data without type conversion
            accumulated_data['label'].append(label.numpy())
            accumulated_data['cls_tokens'].append(cls_tokens.cpu().numpy())
            accumulated_data['visual_tokens'].append(visual_tokens.cpu().numpy())

            if (batch_index + 1) % BATCH_ACCUMULATION == 0:
                # Convert accumulated data to PyArrow arrays
                label_array = pa.array(np.concatenate(accumulated_data['label']))
                cls_tokens_array = pa.array(np.concatenate(accumulated_data['cls_tokens']).tolist())
                visual_tokens_array = pa.array(np.concatenate(accumulated_data['visual_tokens']).tolist())

                # Create a PyArrow Table
                table = pa.Table.from_arrays([label_array, cls_tokens_array, visual_tokens_array],
                                             schema=parquet_schema)

                # Write to Parquet file
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(parquet_file, table.schema)
                parquet_writer.write_table(table)

                # Clear accumulated data
                for key in accumulated_data:
                    accumulated_data[key] = []

        # Write any remaining data
        if any(accumulated_data.values()):
            label_array = pa.array(np.concatenate(accumulated_data['label']))
            cls_tokens_array = pa.array(np.concatenate(accumulated_data['cls_tokens']).tolist())
            visual_tokens_array = pa.array(np.concatenate(accumulated_data['visual_tokens']).tolist())

            table = pa.Table.from_arrays([label_array, cls_tokens_array, visual_tokens_array],
                                         schema=parquet_schema)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_file, table.schema)
            parquet_writer.write_table(table)

    if parquet_writer:
        parquet_writer.close()

# subsets = ['biggan',  'cyclegan',  'dalle_2',  'dalle_mini',  'gaugan',  'glide',  'mj',  'progan',  'sd14',  'sd21', 'stargan',  'stylegan',  'stylegan2']
#
# for subset in subsets:
#     dataset = BinaryJsonDatasets("/home/jwang/ybwork/data/DFBenchmark/DIF_testset", transform=transform, subset=subset, split='test')
#     parquet_file = os.path.join('.', r'clipL14openai_next_to_last_DIF_{}_features.parquet'.format(subset))
#     # dataset = BinaryJsonDatasets("/scratch/yw26g23/datasets/DFBenchmark/DIF_testset", transform=transform, subset=subset, split='test')
#     # parquet_file = os.path.join('/scratch/yw26g23', r'clipL14openai_next_to_last_DIF_{}_features.parquet'.format(subset))
#
#     BATCH_ACCUMULATION = 128
#
#     data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=32)
#
#     model = get_model('clipL14openai_next_to_last')
#     model.cuda()
#     model.to('cuda')
#     model.eval()
#
#     parquet_schema = pa.schema([
#         ('label', pa.int32()),
#         ('cls_tokens', pa.list_(pa.float32())),
#         ('visual_tokens', pa.list_(pa.list_(pa.float32()))),
#     ])
#
#     parquet_writer = None
#     accumulated_tables = []
#     accumulated_data = {
#         'label': [],
#         'cls_tokens': [],
#         'visual_tokens': []
#     }
#
#     with torch.no_grad():
#         print("Length of dataset: %d" % len(data_loader))
#         for batch_index, (img, label) in enumerate(tqdm(data_loader)):
#             in_tens = img.cuda(non_blocking=True)
#             cls_tokens, visual_tokens = model.forward_features(in_tens)
#
#             # Accumulate data without type conversion
#             accumulated_data['label'].append(label.numpy())
#             accumulated_data['cls_tokens'].append(cls_tokens.cpu().numpy())
#             accumulated_data['visual_tokens'].append(visual_tokens.cpu().numpy())
#
#             if (batch_index + 1) % BATCH_ACCUMULATION == 0:
#                 # Convert accumulated data to PyArrow arrays
#                 label_array = pa.array(np.concatenate(accumulated_data['label']))
#                 cls_tokens_array = pa.array(np.concatenate(accumulated_data['cls_tokens']).tolist())
#                 visual_tokens_array = pa.array(np.concatenate(accumulated_data['visual_tokens']).tolist())
#
#                 # Create a PyArrow Table
#                 table = pa.Table.from_arrays([label_array, cls_tokens_array, visual_tokens_array],
#                                              schema=parquet_schema)
#
#                 # Write to Parquet file
#                 if parquet_writer is None:
#                     parquet_writer = pq.ParquetWriter(parquet_file, table.schema)
#                 parquet_writer.write_table(table)
#
#                 # Clear accumulated data
#                 for key in accumulated_data:
#                     accumulated_data[key] = []
#
#         # Write any remaining data
#         if any(accumulated_data.values()):
#             label_array = pa.array(np.concatenate(accumulated_data['label']))
#             cls_tokens_array = pa.array(np.concatenate(accumulated_data['cls_tokens']).tolist())
#             visual_tokens_array = pa.array(np.concatenate(accumulated_data['visual_tokens']).tolist())
#
#             table = pa.Table.from_arrays([label_array, cls_tokens_array, visual_tokens_array],
#                                          schema=parquet_schema)
#
#             if parquet_writer is None:
#                 parquet_writer = pq.ParquetWriter(parquet_file, table.schema)
#             parquet_writer.write_table(table)
#
#     if parquet_writer:
#         parquet_writer.close()

















# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/lsun_bedroom/real', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/lsun_bedroom/ldm', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/lsun_bedroom/midjourney', transform=transform)

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/celebahq/dalle2', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/DiffusionForensics/celebahq/real', transform=transform)

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/ForenSynths/test/deepfake/0_real', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/ForenSynths/test/deepfake/1_fake', transform=transform)

# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/AIGCDetect/stable_diffusion_v_1_5/1_fake', transform=transform)
# dataset = DeepfakeDataset(root_dir='../data/DFBenchmark/AIGCDetect/stable_diffusion_v_1_5/0_real', transform=transform)

# data_loader = DataLoader(dataset, batch_size=512, shuffle=False)

# all_features = {}
# subset_features = []
# y_true = []
# with torch.no_grad():
#     print("Length of dataset: %d" % (len(data_loader)))
#     for img, label in tqdm(data_loader):
#         in_tens = img.cuda()
#         features = model(in_tens, return_feature=True)[1]
#         subset_features.extend(features.tolist())
#         y_true.extend(label.tolist())
#
# subset_features = np.array(subset_features)
# y_true = np.array(y_true)
# all_features['features'] = [subset_features, y_true]


# with open('clip_ForenSynths_fake_features.pkl', 'wb') as file:
#     pickle.dump(all_features, file)
#
# with open('clip_DiffusionForensics_bedroom_mj_features.pkl', 'wb') as file:
#     pickle.dump(all_features, file)

# with open('clip_DiffusionForensics_celebahq_dalle2_features.pkl', 'wb') as file:
#     pickle.dump(all_features, file)

# with open('clip_ForenSynths_deepfake_real_features.pkl', 'wb') as file:
#     pickle.dump(all_features, file)

# with open('clip_AIGCDetect_sd15_fake_features.pkl', 'wb') as file:
#     pickle.dump(all_features, file)

