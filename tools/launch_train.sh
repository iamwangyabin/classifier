#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --account=ecsstaff
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00


export WANDB_MODE="offline"
module load  git/2.38.1
module load cuda/12.2
module load anaconda/py3.10
eval "$(conda shell.bash hook)"
conda init bash
conda activate timm


PYTHONPATH=./  sh hydit/train.sh --index-file dataset/hsr/jsons/hsr.json

