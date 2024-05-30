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

#python test.py --cfg cfgs/test/a_j90.yaml
#python test.py --cfg cfgs/test/a_j90_2.yaml
#python test.py --cfg cfgs/test/a_j90_3.yaml
#python test.py --cfg cfgs/test/a_j90_4.yaml
#python test.py --cfg cfgs/test/a_j90_5.yaml
#python test.py --cfg cfgs/test/a_j90_ab1.yaml
#python test.py --cfg cfgs/test/a_j90_ab2.yaml
#python test.py --cfg cfgs/test/a_j90_ab3.yaml
#python test.py --cfg cfgs/test/a_j90_ab4.yaml
#python test.py --cfg cfgs/test/a_j90_ab5.yaml
#python test.py --cfg cfgs/test/a_j90_ab6.yaml
#python test.py --cfg cfgs/test/a_j90_ab7.yaml

#python test.py --cfg cfgs/test/test_Ojha_224.yaml
#python test.py --cfg cfgs/test/test_NPR_224.yaml
#python test_lgrad.py --cfg cfgs/test/test_LGrad_224.yaml
#python test.py --cfg cfgs/test/test_Garm_224.yaml
#python test.py --cfg cfgs/test/test_freqnet_224.yaml
#python test.py --cfg cfgs/test/test_CNNSpot0.1_224.yaml
# no resize
#python test.py --cfg cfgs/test/nocompre/test_sprompts_224.yaml
#python test.py --cfg cfgs/test/nocompre/test_FreDect_224.yaml
#python test_fusing.py --cfg cfgs/test/nocompre/test_Fusing_224.yaml
#python test_lnp.py --cfg cfgs/test/nocompre/test_LNP_224.yaml
#python test.py --cfg cfgs/test/nocompre/a.yaml

# jpeg 80
#python test.py --cfg cfgs/test/jpg80/test_Ojha_jpg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/test_NPR_jpg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/test_Garm_jpg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/test_freqnet_jpg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/test_CNNSpot0.1_jpg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/test_sprompts_jpg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/test_FreDect_jpg80_224.yaml
#python test_lgrad.py --cfg cfgs/test/jpg80/test_LGrad_jpeg80_224.yaml
#python test_fusing.py --cfg cfgs/test/jpg80/test_Fusing_jpeg80_224.yaml
#python test_lnp.py --cfg cfgs/test/jpg80/test_LNP_jpeg80_224.yaml
#python test.py --cfg cfgs/test/jpg80/a_j80.yaml

# jpeg 70
#python test.py --cfg cfgs/test/jpg70/test_Ojha_jpg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/test_NPR_jpg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/test_Garm_jpg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/test_freqnet_jpg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/test_CNNSpot0.1_jpg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/test_sprompts_jpg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/test_FreDect_jpg70_224.yaml
#python test_lgrad.py --cfg cfgs/test/jpg70/test_LGrad_jpeg70_224.yaml
#python test_fusing.py --cfg cfgs/test/jpg70/test_Fusing_jpeg70_224.yaml
#python test_lnp.py --cfg cfgs/test/jpg70/test_LNP_jpeg70_224.yaml
#python test.py --cfg cfgs/test/jpg70/a_j70.yaml

# jpeg rand
python test.py --cfg cfgs/test/jpgrand/test_Ojha_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/test_NPR_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/test_Garm_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/test_freqnet_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/test_CNNSpot0.1_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/test_sprompts_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/test_FreDect_jpgrand_224.yaml
python test_lgrad.py --cfg cfgs/test/jpgrand/test_LGrad_jpgrand_224.yaml
python test_fusing.py --cfg cfgs/test/jpgrand/test_Fusing_jpgrand_224.yaml
python test_lnp.py --cfg cfgs/test/jpgrand/test_LNP_jpgrand_224.yaml
python test.py --cfg cfgs/test/jpgrand/a_jpegrand.yaml
