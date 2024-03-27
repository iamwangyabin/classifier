import torch
import torch.nn as nn
import copy

from networks.SPrompts.clip.prompt_learner import load_clip_to_cpu, TextEncoder, PromptLearner


class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'


class SliNet_lp(nn.Module):

    def __init__(self):
        super(SliNet_lp, self).__init__()
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_encoder = TextEncoder(clip_model)

        self.l_prompt = PromptLearner(self.cfg, ['real image', 'deepfake image'], self.clip_model)

        self.v_prompt = nn.Linear(768, 10, bias=False)

        for name, param in self.named_parameters():
            param.requires_grad_(False)
            if "l_prompt" in name:
                param.requires_grad_(True)
            if "v_prompt" in name:
                param.requires_grad_(True)

    def forward(self, image, inference=False):
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.image_encoder(image.type(self.dtype), self.v_prompt.weight)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(self.l_prompt(), self.l_prompt.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # return image_features @ text_features.t()
        logits = logit_scale * image_features @ text_features.t()
        if inference:
            prob_fake = torch.div(logits[:, 1], (logits[:, 0] + logits[:, 1]))
            logits = (2 * prob_fake - 1).unsqueeze(-1)
        return logits

