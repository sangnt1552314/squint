#!/bin/bash
#SBATCH --job-name=squint-lift-cube-dr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-lift-cube-dr-%j.out
#SBATCH --time=03:00:00

# Wrist + third-person cameras (two policy views). Options: wrist | third | wrist_third
export SQUINT_CAMERA_TYPE=wrist_third

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

nvidia-smi

python -u train_squint.py \
    --env_id=SO101LiftCube-v1 \
    --exp_name=lift_cube_dr_1500k \
    --total_timesteps=1500000 \
    --remove_overlay \
    --track \
    --wandb_entity=tsangb34-national-university-of-singapore-students-union \
    --wandb_project_name=cs6283 \
    --wandb_group=SQUINT
echo "train_squint.py exit code: $?"
