#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --account=ecsstaff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=60:00:00

export WANDB_MODE="offline"

module load cuda/11.8
module load anaconda/py3.10
eval "$(conda shell.bash hook)"
conda init bash
conda activate timm


python test.py --cfg cfgs/test/a_j90_ab1.yaml
python test.py --cfg cfgs/test/a_j90_ab2.yaml
python test.py --cfg cfgs/test/a_j90_ab3.yaml
python test.py --cfg cfgs/test/a_j90_ab4.yaml
python test.py --cfg cfgs/test/a_j90_ab5.yaml
python test.py --cfg cfgs/test/a_j90_ab6.yaml






