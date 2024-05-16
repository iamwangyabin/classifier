# For general task test, use deepfake trained clip model.

import io
import csv
import argparse
import hydra
import pickle
import sys
import os
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torch.utils.data
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

from utils.util import load_config_with_cli
from utils.network_factory import get_model
from data.general_dataset.imagenet import ImageNet
from data.general_dataset.imagenet_a import ImageNetA
from data.general_dataset.imagenet_v2 import ImageNetV2
from data.general_dataset.imagenet_r import ImageNetR
from data.general_dataset.imagenet_sketch import ImageNetSketch
from networks.SPrompts.arprompts import load_clip_to_cpu, ARPromptLearner


class DatasetWrapper(TorchDataset):

    def __init__(self, data_source, trans):
        self.data_source = data_source.test
        self.trans = T.Compose(trans)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }
        img = Image.open(item.impath).convert("RGB")
        output["img"] = self.trans(img)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = get_model(conf)
    model.cuda()
    model.eval()

    # data_source = ImageNetA(conf)
    # data_source = ImageNetV2(conf)
    # data_source = ImageNetR(conf)
    # data_source = ImageNetSketch(conf)
    # data_source = ImageNet(conf)

    for data_source in [ImageNetA(conf), ImageNetV2(conf), ImageNetR(conf), ImageNetSketch(conf), ImageNet(conf)]:
        print(data_source.dataset_dir)

        clip_model = load_clip_to_cpu(model.cfg)
        token_embedding = clip_model.token_embedding
        prompt_learner = ARPromptLearner(conf, data_source._classnames, clip_model)
        tokenized_prompts = torch.chunk(prompt_learner.tokenized_prompts, 2, dim=0)[0]

        dataset = DatasetWrapper(data_source, conf.DATASET.trsf)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=8,num_workers=1)

        with torch.no_grad():
            y_true_fake, y_pred_fake = [], []
            y_true_real, y_pred_real = [], []
            print("Length of dataset: %d" % (len(data_loader)))
            for data in tqdm(data_loader):
                in_tens = data['img'].cuda()
                labels = data['label'].cuda()
                # pass
                logits_fake, logits_real = model.forward_general_classnames(in_tens, data_source._classnames,
                                                                            token_embedding, tokenized_prompts)

                preds_fake = logits_fake.argmax(1)
                preds_real = logits_real.argmax(1)

                y_pred_fake.extend(preds_fake.cpu().tolist())
                y_true_fake.extend(labels.cpu().tolist())

                y_pred_real.extend(preds_real.cpu().tolist())
                y_true_real.extend(labels.cpu().tolist())

        accuracy_fake = accuracy_score(y_true_fake, y_pred_fake)
        precision_fake = precision_score(y_true_fake, y_pred_fake, average='weighted')
        recall_fake = recall_score(y_true_fake, y_pred_fake, average='weighted')
        f1_fake = f1_score(y_true_fake, y_pred_fake, average='weighted')

        # Calculate evaluation metrics for real logits
        accuracy_real = accuracy_score(y_true_real, y_pred_real)
        precision_real = precision_score(y_true_real, y_pred_real, average='weighted')
        recall_real = recall_score(y_true_real, y_pred_real, average='weighted')
        f1_real = f1_score(y_true_real, y_pred_real, average='weighted')

        print(f"Fake - Accuracy: {accuracy_fake:.4f}")
        print(f"Fake - Precision: {precision_fake:.4f}")
        print(f"Fake - Recall: {recall_fake:.4f}")
        print(f"Fake - F1 Score: {f1_fake:.4f}")

        print(f"Real - Accuracy: {accuracy_real:.4f}")
        print(f"Real - Precision: {precision_real:.4f}")
        print(f"Real - Recall: {recall_real:.4f}")
        print(f"Real - F1 Score: {f1_real:.4f}")
