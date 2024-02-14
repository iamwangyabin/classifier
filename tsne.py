import io
import csv
import argparse
import hydra
import torch.utils.data


from networks.UniversalFakeDetect.clip_models import CLIPModel, CLIPModel_inc
from utils.util import load_config_with_cli, archive_files
from utils.validate import validate
from data.json_datasets import BinaryJsonDatasets













if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    print("Model loaded..")

    model = CLIPModel('ViT-L/14')
    state_dict = torch.load('networks/UniversalFakeDetect/fc_weights.pth', map_location='cpu')
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    all_results = []

    for set_name, source_conf in conf.datasets.source.items():
        data_root = source_conf.data_root
        for subset in source_conf.sub_sets:
            dataset = BinaryJsonDatasets(conf.datasets, data_root, subset, split='test')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                                      num_workers=conf.datasets.loader_workers)
            ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, num_real, num_fake = validate(model, data_loader)
            print(f"{set_name} {subset}")
            print(f"AP: {ap:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([set_name, subset, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres,
                                num_real, num_fake])

    columns = ['dataset', 'sub_set', 'ap', 'r_acc0', 'f_acc0', 'acc0', 'r_acc1', 'f_acc1', 'acc1', 'best_thres',
               'num_real', 'num_fake']
    with open('model_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)