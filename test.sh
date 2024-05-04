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
#
#python test_fusing.py --cfg cfgs/test/test_Fusing_224.yaml
#
#python test.py --cfg cfgs/test/test_Garm_224.yaml

#python test.py --cfg cfgs/test/test_arp.yaml

#python test.py --cfg cfgs/test/test_CNNSpot0.1_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_CNNSpot0.1_jpg100_224.yaml
#
#python test.py --cfg cfgs/test/test_CNNSpot0.1_webp90_224.yaml
#
#python test.py --cfg cfgs/test/test_CNNSpot0.1_webp100_224.yaml

#CUDA_VISIBLE_DEVICES=2 python test_fusing.py --cfg cfgs/test/test_Fusing_jpeg90_224.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/test_Garm_jpg90_224.yaml

#CUDA_VISIBLE_DEVICES=2 python test_lgrad.py --cfg cfgs/test/test_LGrad_jpeg90_224.yaml

CUDA_VISIBLE_DEVICES=1 python test_lnp.py --cfg cfgs/test/test_LNP_jpeg90_224.yaml

CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_freqnet_jpg90_224.yaml
