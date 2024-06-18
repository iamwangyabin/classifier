#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ecsstaff
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
conda activate webui


export COMMANDLINE_ARGS=" --port 9386   --api  --skip-python-version-check --skip-prepare-environment --skip-install --skip-torch-cuda-test"

python_cmd="python"
LAUNCH_SCRIPT="launch.py"

"${python_cmd}" "${LAUNCH_SCRIPT}" "$@"

