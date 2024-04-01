#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --account=ecsstaff
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00

export WANDB_MODE="offline"

module load cuda/11.8
module load anaconda/py3.10
eval "$(conda shell.bash hook)"
conda init bash
conda activate timm


python test.py --cfg cfgs/test_Ojha_224.yaml




#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test_CNNSpot0.1_224.yaml



