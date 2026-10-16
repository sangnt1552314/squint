#!/bin/bash
#SBATCH --job-name=squint-lift-cube-dr-wrist-third-office3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-lift-cube-dr-wrist-third-office3-%j.out
#SBATCH --time=03:00:00

# Lift task: pick up the cube and lift it. 500k steps.

# Wrist + top-down third-person camera (two policy views, obs keys "wrist_rgb" and "third_rgb"): the third camera is
# the RealSense D435 on its stand in front of the arm, looking down at it (office3's pose, SCENES["office3"]).
# Options: wrist | third | wrist_third. Deploy with the same value (real cameras: WRIST_CAMERA_ID / THIRD_CAMERA_ID
# in deploy_utils/robot_config.py).
export SQUINT_CAMERA_TYPE=wrist_third
# Office3 rig: wrist mount fitted to the office3 captures, office3 room (desk, wall, floor, D435 stand camera).
# Values live in WRIST_CAMERA_MOUNTS["office3"] and SCENES["office3"] (envs/base_random_env.py).
export SQUINT_WRIST_MOUNT=office3
export SQUINT_SCENE=office3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

nvidia-smi
echo "SQUINT_CAMERA_TYPE=${SQUINT_CAMERA_TYPE} SQUINT_WRIST_MOUNT=${SQUINT_WRIST_MOUNT} SQUINT_SCENE=${SQUINT_SCENE}"

# Table top: random color per env (any RGB)
# Shadows: angled light casts shadows, cameras use the slower "default" shader (the office3 colors are matched to it)
# Spawn: a 12 x 12 cm square centred 18.5 cm in front of the base and 7 cm to its right (x 0.125-0.245, y -0.13..-0.01),
# the part of the desk the D435's square crop sees in full (the default square, x 0.2-0.4, is mostly below its image)
python -u train_squint.py \
    --env_id=SO101LiftCube-v1 \
    --exp_name=lift_cube_dr_wrist_third_office3_table_random_shadows_500k \
    --total_timesteps=500000 \
    --spawn_box_pos 0.185 -0.07 \
    --spawn_box_half_size=0.06 \
    --remove_overlay \
    --table_color=random \
    --shadows \
    --track \
    --wandb_entity=tsangb34-national-university-of-singapore-students-union \
    --wandb_project_name=cs6283 \
    --wandb_group=SQUINT
echo "train_squint.py exit code: $?"
