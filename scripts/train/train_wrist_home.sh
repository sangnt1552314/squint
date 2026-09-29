#!/bin/bash
#SBATCH --job-name=squint-lift-cube-dr-wrist-home
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-lift-cube-dr-wrist-home-%j.out
#SBATCH --time=04:00:00

# Wrist camera ONLY (single policy view, obs key "rgb").
# Options: wrist | third | wrist_third. Deploy with the same value.
export SQUINT_CAMERA_TYPE=wrist
# Home SO-101 wrist mount: camera sees the fixed jaw in the lower-left of the frame.
# Values live in WRIST_CAMERA_MOUNTS["home"] (envs/base_random_env.py), measured with tune_camera.py.
export SQUINT_WRIST_MOUNT=home

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

nvidia-smi
echo "SQUINT_CAMERA_TYPE=${SQUINT_CAMERA_TYPE} SQUINT_WRIST_MOUNT=${SQUINT_WRIST_MOUNT}"

# Table top: random color per env (any RGB)
# Shadows: angled light casts shadows, cameras use the slower "default" shader
# Robot color: white, to match the white-printed home SO-101 (training default is matte black)
python -u train_squint.py \
    --env_id=SO101LiftCube-v1 \
    --exp_name=lift_cube_dr_wrist_home_table_random_shadows_1500k \
    --total_timesteps=1500000 \
    --remove_overlay \
    --table_color=random \
    --shadows \
    --robot_color=white \
    --track \
    --wandb_entity=tsangb34-national-university-of-singapore-students-union \
    --wandb_project_name=cs6283 \
    --wandb_group=SQUINT
echo "train_squint.py exit code: $?"
