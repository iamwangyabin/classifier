#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --account=ecsstaff
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00


export WANDB_MODE="offline"

module load cuda/11.8
module load anaconda/py3.10
eval "$(conda shell.bash hook)"
conda init bash
conda activate timm


#python train.py --cfg cfgs/train_coop_progan.yaml
#
#python train.py --cfg cfgs/train_coop_progan_b.yaml

#python train.py --cfg cfgs/train_vlp_progan.yaml
#
#python train.py --cfg cfgs/train_vlp_progan_b.yaml

#python train.py --cfg cfgs/train_clip_sd15.yaml

python train.py --cfg cfgs/train/train_arp_progan.yaml

#python train.py --cfg cfgs/train/train_arp_progan2.yaml
