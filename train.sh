#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --account=ecsstaff
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00


export WANDB_MODE="offline"
export WANDB_API_KEY="a4d3a740e939973b02ac59fbd8ed0d6a151df34b"

module load cuda/11.8
module load anaconda/py3.10
eval "$(conda shell.bash hook)"
conda init bash
conda activate timm


python train.py --cfg cfgs/train/cnndet01.yaml


