#!/bin/bash
#SBATCH --job-name=squint-black
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-black-%j.out
#SBATCH --time=01:30:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

nvidia-smi

python train_squint.py \
    --env_id=SO101LiftBlueCube-v1 \
    --exp_name=lift_blue_cube_1500k \
    --total_timesteps=1500000 \
    --remove_overlay \
    --track \
    --wandb_entity=tsangb34-national-university-of-singapore-students-union \
    --wandb_project_name=cs6283 \
    --wandb_group=SQUINT