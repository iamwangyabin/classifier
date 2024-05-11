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

#python test.py --cfg cfgs/test/test_CNNSpot0.1_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_CNNSpot0.5_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_FreDect_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_NPR_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_Ojha_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_sprompts_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_freqnet_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_Garm_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_arp_jpg90_224.yaml
#
#python test.py --cfg cfgs/test/test_arp2_jpg90_224.yaml

python train.py --cfg cfgs/train/train_arp_progan.yaml

python train.py --cfg cfgs/train/train_arp_progan2.yaml

python train.py --cfg cfgs/train/train_arp_progan3.yaml

python train.py --cfg cfgs/train/train_arp_progan4.yaml

python train.py --cfg cfgs/train/train_arp_progan5.yaml

python train.py --cfg cfgs/train/train_arp_progan6.yaml

python train.py --cfg cfgs/train/train_arp_progan7.yaml

#python test_fusing.py --cfg cfgs/test/test_Fusing_jpeg90_224.yaml

##python test_lgrad.py --cfg cfgs/test/test_LGrad_jpeg90_224.yaml

#python test_lnp.py --cfg cfgs/test/test_LNP_jpeg90_224.yaml


#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f_6.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_j90.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_j90_2.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f_3.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_j90_4.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f_5.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f_4.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f_2.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_f_7.yaml

#CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test/a_j90_3.yaml

#python test.py --cfg cfgs/test/a_j90.yaml
#python test.py --cfg cfgs/test/a_j90_2.yaml
#python test.py --cfg cfgs/test/a_j90_3.yaml
#python test.py --cfg cfgs/test/a_j90_4.yaml
#python test.py --cfg cfgs/test/a_f.yaml
#python test.py --cfg cfgs/test/a_f_2.yaml
#python test.py --cfg cfgs/test/a_f_3.yaml
#python test.py --cfg cfgs/test/a_f_4.yaml
#python test.py --cfg cfgs/test/a_f_5.yaml
#python test.py --cfg cfgs/test/a_f_6.yaml













