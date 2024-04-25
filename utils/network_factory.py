import torch
import timm
import torchvision

from networks.resnet import resnet50
from networks.UniversalFakeDetect.clip_models import CLIPModel, CLIPModel_inc
from networks.MultiscaleCLIP.clip_models import MultiscaleCLIPModel
from networks.DINO.detector import DINOModel
from networks.NPR.detector import NPRModel
from networks.ViTDetector.detector import ViTModel
from networks.SPrompts.independentVL import IndepVLPCLIP
from networks.SPrompts.coop import CoOpCLIP
from networks.SPrompts.arprompts import ARPromptsCLIP

def resume_lightning(model, conf):
    if conf.resume:
        # for ojha fc weight
        # state_dict = torch.load(conf.resume, map_location='cpu')
        # model.fc.load_state_dict(state_dict)
        state_dict = torch.load(conf.resume, map_location='cpu')['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # remove `model.` from key
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)


def get_model(conf):
    print("Model loaded..")
    if conf.arch == 'clip':
        model = CLIPModel('ViT-L/14')
        state_dict = torch.load(conf.resume, map_location='cpu')
        model.fc.load_state_dict(state_dict)
        # resume_lightning(model, conf)

    elif conf.arch == 'cnn':
        model = resnet50(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif conf.arch == 'npr':
        model = NPRModel(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            try:
                model.load_state_dict(state_dict['model'])
            except:
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
    elif conf.arch == 'lnp':
        model = resnet50(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.load_state_dict(state_dict['model'])
    elif conf.arch == 'freqnet':
        from networks.FreqNet.freqnet import freqnet
        model = freqnet(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.load_state_dict(state_dict)

    # elif conf.arch == 'LGrad':
    #     model = resnet50(num_classes=1)
    #     if conf.resume:

    elif conf.arch == 'FreDect':
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048, 1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            try:
                model.load_state_dict(state_dict['netC'], strict=True)
            except:
                model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})

    elif conf.arch == 'Fusing':
        from networks.Fusing.detector import Patch5Model
        model = Patch5Model()
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            model.load_state_dict(state_dict['model'])

    elif conf.arch == 'Gram':
        from networks.GramNet.detector import resnet18
        model = resnet18(num_classes=1)
        if conf.resume:
            state_dict = torch.load(conf.resume, map_location='cpu')
            try:
                model.load_state_dict(state_dict['netC'], strict=True)
            except:
                model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()})

    elif conf.arch == 'arp':
        model = ARPromptsCLIP(conf)
        resume_lightning(model, conf)

    # elif conf.arch == 'dino':
    #     model = DINOModel('dinov2_l')
    #     resume_lightning(model, conf)
    #
    # elif conf.arch == 'vlp':
    #     model = IndepVLPCLIP(conf)
    #     resume_lightning(model, conf)
    #
    # elif conf.arch == 'coop':
    #     model = CoOpCLIP(conf)
    #     resume_lightning(model, conf)
    # elif conf.arch == 'clip_res':
    #     model = CLIPModel('RN50x64')
    # elif conf.arch == 'clip_vit336':
    #     model = CLIPModel('ViT-L/14@336px')
    # elif conf.arch == 'vit_l':
    #     model = ViTModel('vit_large_patch16_224')
    # elif conf.arch == 'ms_clip':
    #     model = MultiscaleCLIPModel('ViT-L/14', patch_sizes=conf.patch_sizes, strides=conf.strides)
    #     if conf.resume:
    #         state_dict = torch.load(conf.resume, map_location='cpu')
    #         model.fc.load_state_dict(state_dict)

    else:
        model = timm.create_model(conf.arch, pretrained=True, num_classes=1)
        torch.nn.init.xavier_uniform_(model.head.weight.data)

    return model

