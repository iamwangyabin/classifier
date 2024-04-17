#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ecsstaff
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


#python test.py --cfg cfgs/test_coop.yaml
#
#python test.py --cfg cfgs/test_coop2.yaml
#
#python test.py --cfg cfgs/test_vlp.yaml
#
#python test.py --cfg cfgs/test_vlp2.yaml

python test.py --cfg cfgs/test/test_arp.yaml





