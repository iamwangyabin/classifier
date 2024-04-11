# plz install https://github.com/marcojira/fld.git

import os
import csv
import json
import hydra
import argparse
from PIL import ImageFile, Image
from utils.util import load_config_with_cli
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_fid.inception import InceptionV3

from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
# from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.features.ImageFeatureExtractor import ImageFeatureExtractor
from fld.metrics.FID import FID
from fld.metrics.FLD import FLD
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall
from torchmetrics.image.fid import FrechetInceptionDistance

# from cleanfid import fid
# score = fid.compute_fid("/home/jwang/ybwork/data/deepfake_benchmark/DiffusionForensics/lsun_bedroom/real", "/home/jwang/ybwork/data/deepfake_benchmark/DiffusionForensics/lsun_bedroom/sdv1_new1")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JsonDatasets(Dataset):
    def __init__(self, opt, data_root, subset='all', split='train', selected_label=0, transform=None):
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
            if label == selected_label:
                self.image_pathes.append(img_full_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label



def fisher_score(feature1, feature2):
    mean_A = torch.mean(feature1, dim=0)
    mean_B = torch.mean(feature2, dim=0)

    def within_class_scatter(feat, mean):
        diff = feat - mean
        scatter = diff.T @ diff
        return scatter

    scatter_A = within_class_scatter(feature1, mean_A)
    scatter_B = within_class_scatter(feature2, mean_B)
    S_W = scatter_A + scatter_B
    mean_diff = (mean_A - mean_B).reshape(-1, 1)
    S_B = mean_diff @ mean_diff.T
    S_W_inv = torch.pinverse(S_W)
    w = S_W_inv @ mean_diff
    return w



class InceptionFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None):
        self.name = "inception"

        super().__init__(save_path)

        self.features_size = 2048
        self.preprocess = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((299,299)),
            transforms.ToTensor(),])

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3(
            [block_idx], resize_input=True, normalize_input=True
        ).to(DEVICE)
        self.model.eval()
        return

    def get_feature_batch(self, img_batch):
        assert img_batch.max() <= 1
        assert img_batch.min() >= 0
        with torch.no_grad():
            features = self.model(img_batch)[0]   #
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()
        return features



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    all_results = []
    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        json_file = os.path.join(data_root, 'test.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        for subset in source_conf.sub_sets:
            try:
                print(f"{set_name} {subset}")
                clip_feature_extractor = CLIPFeatureExtractor()
                dino_feature_extractor = DINOv2FeatureExtractor()
                inception_feature_extractor = InceptionFeatureExtractor()

                gen_dataset = JsonDatasets(conf, data_root, subset, split='test', selected_label=1)
                real_dataset = JsonDatasets(conf, data_root, subset, split='test', selected_label=0)

                # fid = FrechetInceptionDistance(feature=64)
                # trans = transforms.Compose([
                #     transforms.Resize((256, 256)),
                #     transforms.ToTensor(),
                # ])
                # gen_dataset2 = JsonDatasets(conf, data_root, subset, split='test', selected_label=1, transform=trans)
                # real_dataset2 = JsonDatasets(conf, data_root, subset, split='test', selected_label=0, transform=trans)
                # gen_dataloader = DataLoader(gen_dataset2, batch_size=256, num_workers=8)
                # real_dataloader = DataLoader(real_dataset2, batch_size=256, num_workers=8)
                # for batch in tqdm(gen_dataloader):
                #     images = (batch[0] * 255).to(dtype=torch.uint8)
                #     fid.update(images, real=False)
                # for batch in tqdm(real_dataloader):
                #     images = (batch[0] * 255).to(dtype=torch.uint8)
                #     fid.update(images, real=True)
                # fid = fid.compute().item()
                # print(fid)
                # kid = 0

                inception_gen_feat = inception_feature_extractor.get_features(gen_dataset)
                inception_real_feat = inception_feature_extractor.get_features(real_dataset)

                target_feature_count = 2048

                if inception_gen_feat.size(0) < target_feature_count:
                    repeat_times = target_feature_count // inception_gen_feat.size(0)
                    additional_copies = repeat_times - 1
                    if additional_copies > 0:
                        inception_gen_feat = torch.cat([inception_gen_feat] * (additional_copies + 1), dim=0)
                    inception_gen_feat = inception_gen_feat[:target_feature_count]

                if inception_real_feat.size(0) < target_feature_count:
                    repeat_times = target_feature_count // inception_real_feat.size(0)
                    additional_copies = repeat_times - 1
                    if additional_copies > 0:
                        inception_real_feat = torch.cat([inception_real_feat] * (additional_copies + 1), dim=0)
                    inception_real_feat = inception_real_feat[:target_feature_count]
                try:
                    fid = FID().compute_metric(inception_real_feat, None, inception_gen_feat)
                except:
                    print("faild fid")
                    fid = 0
                try:
                    kid = KID().compute_metric(inception_real_feat, None, inception_gen_feat)
                except:
                    print("faild kid")
                    kid = 0

                clip_gen_feat = clip_feature_extractor.get_features(gen_dataset)
                clip_real_feat = clip_feature_extractor.get_features(real_dataset)

                dino_gen_feat = dino_feature_extractor.get_features(gen_dataset)
                dino_real_feat = dino_feature_extractor.get_features(real_dataset)


                try:
                    clip_fid = FID().compute_metric(clip_real_feat, None, clip_gen_feat)
                except:
                    clip_fid = 0

                try:
                    clip_kid = KID().compute_metric(clip_real_feat, None, clip_gen_feat)
                except:
                    clip_kid=0
                clip_precision = PrecisionRecall(mode="Precision").compute_metric(clip_real_feat, None, clip_gen_feat)
                clip_reacall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(clip_real_feat, None, clip_gen_feat)

                try:
                    dino_fid = FID().compute_metric(dino_real_feat, None, dino_gen_feat)
                except:
                    dino_fid = 0

                try:
                    dino_kid = KID().compute_metric(dino_real_feat, None, dino_gen_feat)
                except:
                    dino_kid = 0

                dino_precision = PrecisionRecall(mode="Precision").compute_metric(dino_real_feat, None, dino_gen_feat)
                dino_reacall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(dino_real_feat, None, dino_gen_feat)

                minsize = min(clip_real_feat.size(0), clip_gen_feat.size(0), dino_real_feat.size(0), dino_gen_feat.size(0))
                clip_score = F.cosine_similarity(clip_real_feat[:minsize, :], clip_gen_feat[:minsize, :], dim=1).mean().item()
                dino_score = F.cosine_similarity(dino_real_feat[:minsize, :], dino_gen_feat[:minsize, :], dim=1).mean().item()

                # fisher_score(clip_gen_feat, clip_real_feat)

                all_results.append([set_name, subset, fid, kid, clip_fid, clip_kid, clip_precision, clip_reacall, dino_fid, dino_kid, dino_precision, dino_reacall, clip_score, dino_score])

                # inception_precision = PrecisionRecall(mode="Precision").compute_metric(inception_real_feat, None, inception_gen_feat)
                # inception_reacall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(inception_real_feat, None, inception_gen_feat)

                # score = 100 * (clip_real_feat * clip_gen_feat).sum(axis=-1)

                columns = ['dataset', 'sub_set', 'fid', 'kid', 'clip_fid', 'clip_kid', 'clip_precision', 'clip_reacall', 'dino_fid', 'dino_kid', 'dino_precision', 'dino_reacall', 'clip_score', 'dino_score']
                with open('dataset_analysis_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    for values in all_results:
                        writer.writerow(values)

            except:
                all_results.append([set_name, subset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                print('failed')



