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


#python train.py --cfg cfgs/train/train_arp_progan.yaml
#python train.py --cfg cfgs/train/train_arp_progan2.yaml
#python train.py --cfg cfgs/train/train_arp_progan3.yaml
#python train.py --cfg cfgs/train/train_arp_progan4.yaml
#python train.py --cfg cfgs/train/train_arp_progan5.yaml
#python train.py --cfg cfgs/train/train_arp_progan6.yaml
#python train.py --cfg cfgs/train/train_arp_progan7.yaml
#python train.py --cfg cfgs/train/train_arp_progan8.yaml
#python train.py --cfg cfgs/train/train_arp_progan9.yaml
#python train.py --cfg cfgs/train/train_arp_progan10.yaml
#python train.py --cfg cfgs/train/train_arp_progan11.yaml
#python train.py --cfg cfgs/train/train_arp_progan12.yaml
#python train.py --cfg cfgs/train/train_arp_progan13.yaml

#python train.py --cfg cfgs/train/train_arp_progan6.yaml
#
#python train.py --cfg cfgs/train/train_arp_progan7.yaml

#python test_fusing.py --cfg cfgs/test/test_Fusing_jpeg90_224.yaml

##python test_lgrad.py --cfg cfgs/test/test_LGrad_jpeg90_224.yaml

#python test_lnp.py --cfg cfgs/test/test_LNP_jpeg90_224.yaml

python test.py --cfg cfgs/test/a_j90.yaml
python test.py --cfg cfgs/test/a_j90_2.yaml
python test.py --cfg cfgs/test/a_j90_3.yaml
python test.py --cfg cfgs/test/a_j90_4.yaml
python test.py --cfg cfgs/test/a_j90_5.yaml
python test.py --cfg cfgs/test/a_j90_ab1.yaml
python test.py --cfg cfgs/test/a_j90_ab2.yaml
python test.py --cfg cfgs/test/a_j90_ab3.yaml
python test.py --cfg cfgs/test/a_j90_ab4.yaml
python test.py --cfg cfgs/test/a_j90_ab5.yaml
python test.py --cfg cfgs/test/a_j90_ab6.yaml
python test.py --cfg cfgs/test/a_j90_ab7.yaml














