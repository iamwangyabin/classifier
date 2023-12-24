import os
import csv
import torch
import torch.nn as nn
import timm

from utils.validate import validate

from options.test_options import TestOptions


# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)

    opt.isTrain = False
    opt.no_resize = False
    opt.no_crop = False
    opt.serial_batches = True
    opt.jpg_method = ['pil']

    print(opt.dataroot)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    # opt.no_resize = True    # testing without resizing by default

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
    # model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))



csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
