import io
import csv
import argparse
import hydra
import torch.utils.data
import pickle
from tqdm import tqdm
import numpy as np

from utils.util import load_config_with_cli, archive_files
from utils.validate import validate
from data.json_datasets import BinaryJsonDatasets
from data.binary_datasets import BinaryMultiDatasets
from utils.network_factory import get_model

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

    with open('clip_progan_train_features.pkl', 'wb') as file:
        pickle.dump(all_features, file)


    #
    # all_features = {}
    # for set_name, source_conf in conf.datasets.source.items():
    #     data_root = source_conf.data_root
    #     for subset in source_conf.sub_sets:
    #         # dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
    #         dataset = BinaryMultiDatasets(conf.dataset.train, split='train')
    #         data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
    #                                                   num_workers=conf.datasets.loader_workers)
    #         subset_features = []
    #         y_true = []
    #         with torch.no_grad():
    #             print("Length of dataset: %d" % (len(data_loader)))
    #             for img, label in tqdm(data_loader):
    #                 in_tens = img.cuda()
    #                 features = model(in_tens, return_feature=True)[1]
    #                 subset_features.extend(features.tolist())
    #                 y_true.extend(label.tolist())
    #         subset_features = np.array(subset_features)
    #         y_true = np.array(y_true)
    #         all_features[f"{set_name} {subset}"] = [subset_features, y_true]
    #
    # with open(conf.test_name + 'features.pkl', 'wb') as file:
    #     pickle.dump(all_features, file)
