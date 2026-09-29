#!/bin/bash
#SBATCH --job-name=squint-stack-cube-dr-wrist-office
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-stack-cube-dr-wrist-office-%j.out
#SBATCH --time=04:00:00

# Stack task: pick the red cube (itemA, 1-5 cm) and place it on the larger blue cube (itemB, 4-6 cm).
# Sizes live in StackRandomizationConfig (envs/stack.py).

# Wrist camera ONLY (single policy view, obs key "rgb").
# Options: wrist | third | wrist_third. Deploy with the same value.
export SQUINT_CAMERA_TYPE=wrist
# Office SO-101 wrist mount: camera sees the fixed jaw in the lower-left of the frame.
# Values live in WRIST_CAMERA_MOUNTS["office"] (envs/base_random_env.py), measured with tune_camera.py.
export SQUINT_WRIST_MOUNT=office

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

nvidia-smi
echo "SQUINT_CAMERA_TYPE=${SQUINT_CAMERA_TYPE} SQUINT_WRIST_MOUNT=${SQUINT_WRIST_MOUNT}"

# Table top: random color per env (any RGB)
# Shadows: angled light casts shadows, cameras use the slower "default" shader
# Item colors: kept fixed (red itemA, blue itemB); set --randomize_item_color=True for random colors
python -u train_squint.py \
    --env_id=SO101StackCube-v1 \
    --exp_name=stack_cube_dr_wrist_office_table_random_shadows_1500k \
    --total_timesteps=1500000 \
    --remove_overlay \
    --table_color=random \
    --shadows \
    --randomize_item_color=False \
    --track \
    --wandb_entity=tsangb34-national-university-of-singapore-students-union \
    --wandb_project_name=cs6283 \
    --wandb_group=SQUINT
echo "train_squint.py exit code: $?"
