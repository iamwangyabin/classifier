# plz install https://github.com/marcojira/fld.git

import os
import csv
import json
import hydra
import argparse
from PIL import ImageFile, Image
from utils.util import load_config_with_cli

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FID import FID
from fld.metrics.FLD import FLD
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall


class JsonDatasets(Dataset):
    def __init__(self, opt, data_root, subset='all', split='train', selected_label=0):
        self.dataroot = data_root
        self.split = split
        self.image_pathes = []
        self.labels = []

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

                inception_feature_extractor.preprocess = transforms.Compose(
                        [transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
                         transforms.CenterCrop(299),
                         transforms.ToTensor(),]
                    )

                inception_gen_feat = inception_feature_extractor.get_features(gen_dataset)
                inception_real_feat = inception_feature_extractor.get_features(real_dataset)


                clip_gen_feat = clip_feature_extractor.get_features(gen_dataset)
                clip_real_feat = clip_feature_extractor.get_features(real_dataset)


                dino_gen_feat = dino_feature_extractor.get_features(gen_dataset)
                dino_real_feat = dino_feature_extractor.get_features(real_dataset)


                clip_fid = FID().compute_metric(clip_real_feat, None, clip_gen_feat)
                clip_kid = KID().compute_metric(clip_real_feat, None, clip_gen_feat)
                clip_precision = PrecisionRecall(mode="Precision").compute_metric(clip_real_feat, None, clip_gen_feat)
                clip_reacall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(clip_real_feat, None, clip_gen_feat)

                dino_fid = FID().compute_metric(dino_real_feat, None, dino_gen_feat)
                dino_kid = KID().compute_metric(dino_real_feat, None, dino_gen_feat)
                dino_precision = PrecisionRecall(mode="Precision").compute_metric(dino_real_feat, None, dino_gen_feat)
                dino_reacall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(dino_real_feat, None, dino_gen_feat)


                minsize = min(clip_real_feat.size(0), clip_gen_feat.size(0), dino_real_feat.size(0), dino_gen_feat.size(0))
                clip_score = F.cosine_similarity(clip_real_feat[:minsize, :], clip_gen_feat[:minsize, :], dim=1).mean().item()
                dino_score = F.cosine_similarity(dino_real_feat[:minsize, :], dino_gen_feat[:minsize, :], dim=1).mean().item()

                # fisher_score(clip_gen_feat, clip_real_feat)
                fid = FID().compute_metric(inception_real_feat, None, inception_gen_feat)
                kid = KID().compute_metric(inception_real_feat, None, inception_gen_feat)

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



















