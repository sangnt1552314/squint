#!/bin/bash
#SBATCH --job-name=squint-lift-cube-dr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-lift-cube-dr-%j.out
#SBATCH --time=01:30:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

nvidia-smi

# python train_squint.py \
#     --env_id=SO101LiftCube-v1 \
#     --exp_name=lift_cube_dr_1500k \
#     --num_envs=16 \
#     --num_eval_envs=4 \
#     --total_timesteps=1500000 \
#     --remove_overlay \
#     --track \
#     --wandb_entity=tsangb34-national-university-of-singapore-students-union \
#     --wandb_project_name=cs6283 \
#     --wandb_group=SQUINT

python train_squint.py --env_id=SO101LiftCube-v1 --exp_name=smoke \
    --num_envs=16 --num_eval_envs=4 --total_timesteps=2000 --learning_starts=500 \
    --eval_freq=1000 --remove_overlay --no-compile --no-cudagraphs
