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


#python test.py --cfg cfgs/test/test_CNNSpot0.1_224.yaml
#
#python test.py --cfg cfgs/test/test_CNNSpot0.5_224.yaml
#
#python test.py --cfg cfgs/test/test_Ojha_224.yaml
#
#python test.py --cfg cfgs/test/test_FreDect_224.yaml


#python test_fusing.py --cfg cfgs/test/test_Fusing_224.yaml
#
#python test.py --cfg cfgs/test/test_Garm_224.yaml


#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/test_NPR_224.yaml
#
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/test_reimpleNPR_224.yaml
#
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/test_freqnet_224.yaml


#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/test_reimplefreqnet_224.yaml

#python test.py --cfg cfgs/test/test_arp.yaml

python test.py --cfg cfgs/test/test_CNNSpot0.5_jpg90_224.yaml

python test.py --cfg cfgs/test/test_CNNSpot0.5_jpg100_224.yaml

python test.py --cfg cfgs/test/test_CNNSpot0.5_webp90_224.yaml

python test.py --cfg cfgs/test/test_CNNSpot0.5_webp100_224.yaml