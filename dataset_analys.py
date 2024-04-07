import argparse
from utils.util import load_config_with_cli
import os
import hydra
import json
from accelerate import Accelerator
from PIL import ImageFile, Image

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.metrics.clip_score import CLIPIScore as CLIP_IScore
from utils.metrics.dino_score import DINOScore as DINO_Score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    all_results = []
    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            json_file = os.path.join(source_conf.data_root, 'test.json')
            with open(json_file, 'r') as f:
                data = json.load(f)

            accelerator = Accelerator()

            dino_score = DINO_Score(model_name_or_path='dino_vits16')
            clip_i_score = CLIP_IScore(model_name_or_path='openai/clip-vit-base-patch32')
            fid = FrechetInceptionDistance(feature=2048)
            inception = InceptionScore()

            dino_score = accelerator.prepare_model(dino_score, evaluation_mode=True)
            clip_i_score = accelerator.prepare_model(clip_i_score, evaluation_mode=True)
            fid = accelerator.prepare_model(fid, evaluation_mode=True)
            inception = accelerator.prepare_model(inception, evaluation_mode=True)


            fake_images, real_images = [], []
            for img_rel_path, label in data[subset].items():
                img_full_path = os.path.join(source_conf.data_root, img_rel_path)
                if label == 1:
                    fake_images.append(Image.open(img_full_path).convert('RGB'))
                else:
                    real_images.append(Image.open(img_full_path).convert('RGB'))


            dino_score.update(fake_images, real_images)
            clip_i_score.update(fake_images, real_images)
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)
            inception.update(fake_images)

            print(f"{set_name} {subset}")
            all_results.append([set_name, subset, (dino_score.compute()).item(), (clip_i_score.compute()).item(),
                                (fid.compute()).item(), (inception.compute()).item()])


            import pdb;pdb.set_trace()














