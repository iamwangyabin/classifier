import torch

from networks.resnet import resnet50
from networks.UniversalFakeDetect.clip_models import CLIPModel, CLIPModel_inc
from networks.MultiscaleCLIP.clip_models import MultiscaleCLIPModel
from networks.DINO.detector import DINOModel
from networks.SPrompts.slinet import SliNet_lp

def get_model(conf):
    print("Model loaded..")
    if conf.arch == 'clip':
        model = CLIPModel('ViT-L/14')
        if conf.resume:
            # for ojha fc weight
            # state_dict = torch.load(conf.resume, map_location='cpu')
            # model.fc.load_state_dict(state_dict)
            state_dict = torch.load(conf.resume, map_location='cpu')['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:] # remove `model.` from key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)

    elif conf.arch == 'ms_clip':
        model = MultiscaleCLIPModel('ViT-L/14', patch_sizes=conf.patch_sizes, strides=conf.strides)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.fc.load_state_dict(state_dict)
    elif conf.arch == 'cnn':
        model = resnet50(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif conf.arch == 'lnp':
        model = resnet50(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif conf.arch == 'dino':
        model = DINOModel('dinov2_l')
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:] # remove `model.` from key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)

    elif conf.arch == 'sp_l':
        model = SliNet_lp()
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)

    elif conf.arch == 'clip_res':
        model = CLIPModel('RN50x64')
    elif conf.arch == 'clip_vit336':
        model = CLIPModel('ViT-L/14@336px')
    else:
        model = timm.create_model(conf.arch, pretrained=True, num_classes=1)
        torch.nn.init.xavier_uniform_(model.head.weight.data)

    return model

