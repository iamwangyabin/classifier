import torch
import torch.nn as nn
import copy

from networks.SPrompts.clip import load
from networks.SPrompts.textprompt_learner import PromptLearner
class cfgc(object):
    backbonename = 'ViT-B/16'
    NCTX = 16
    CTXINIT = ''
    CSC = False
    CLASS_TOKEN_POSITION = 'end'


# AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
#                           "learnabel_text_embedding_length": args.t_n_ctx}






class PromptCLIP(nn.Module):

    def __init__(self):
        super(PromptCLIP, self).__init__()
        model, _ = load("ViT-L/14@336px", device=device)
        model.eval()
        prompt_learner = PromptLearner(model.to("cpu"), configs)
        self.dtype = model.dtype

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

