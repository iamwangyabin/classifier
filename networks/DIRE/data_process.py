"""
Modified from guided-diffusion/scripts/image_sample.py
"""
import argparse
import os
import torch
import sys
import cv2
from mpi4py import MPI

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        timestep_respacing='ddim20',
        batch_size=16,
        use_ddim=True,
        model_path="",
        real_step=0,
        continue_reverse=False,
        has_subfolder=True,
        attention_resolutions="32,16,8",
        class_cond=False,
        dropout=0.1,
        diffusion_steps=1000,
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.convert_to_fp16()
    model.eval()

    # data = load_data_for_reverse(
    #     data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    # )

    imgs = imgs[:batch_size]
    imgs = imgs.to(dist_util.dev())

    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())
        model_kwargs["y"] = classes
    reverse_fn = diffusion.ddim_reverse_sample_loop
    imgs = reshape_image(imgs, args.image_size)

    latent = reverse_fn(
        model,
        (batch_size, 3, args.image_size, args.image_size),
        noise=imgs,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        real_step=args.real_step,
    )
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    recons = sample_fn(
        model,
        (batch_size, 3, args.image_size, args.image_size),
        noise=latent,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        real_step=args.real_step,
    )

    dire = th.abs(imgs - recons)

    dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
    dire = dire.permute(0, 2, 3, 1)
    dire = dire.contiguous()



