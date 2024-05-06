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

#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_FreDect_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_sprompts_face_224.yaml
#CUDA_VISIBLE_DEVICES=2 python test_lgrad.py --cfg cfgs/test/test_LGrad_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_NPR_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_freqnet_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_CNNSpot0.5_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_Garm_face_224.yaml
#CUDA_VISIBLE_DEVICES=2 python test_lnp.py --cfg cfgs/test/test_LNP_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_Ojha_face_224.yaml
CUDA_VISIBLE_DEVICES=1 python test_fusing.py --cfg cfgs/test/test_Fusing_face_224.yaml
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test/test_CNNSpot0.1_face_224.yaml