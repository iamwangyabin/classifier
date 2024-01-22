import os
import csv
import torch
import torch.nn as nn
import timm

from utils.validate import validate
from options.test_options import TestOptions







if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    opt.isTrain = False
    opt.no_resize = False
    opt.no_crop = False
    opt.serial_batches = True
    opt.jpg_method = ['pil']
    dataroot = opt.dataroot


    model_name = os.path.basename(opt.model_path).replace('.pth', '')
    rows = [["{} model testing on...".format(model_name)], ['testset', 'accuracy', 'avg precision']]
    print("{} model testing on...".format(model_name))

    sub_dirs = opt.sub_dirs.split(',')

    for v_id, val in enumerate(sub_dirs):
        opt.dataroot = '{}/{}'.format(dataroot, val)

        model = timm.create_model(opt.arch, pretrained=False, num_classes=1)
        state_dict = torch.load(opt.model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model.cuda()
        model.eval()

        acc, ap, _, _, _, _ = validate(model, opt)
        rows.append([val, acc, ap])
        print("({}) acc: {}; ap: {}".format(val, acc, ap))

    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    csv_name = opt.results_dir + '/{}.csv'.format(model_name)
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)
