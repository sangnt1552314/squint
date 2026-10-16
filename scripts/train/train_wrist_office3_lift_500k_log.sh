#!/bin/bash
#SBATCH --job-name=squint-lift-cube-dr-wrist-office3-log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --output=squint-lift-cube-dr-wrist-office3-log-%j.out
#SBATCH --time=02:00:00

# Lift task: pick up the cube and lift it. 500k steps.
# Same as train_wrist_office3_lift_500k.sh, but the log shows what is running: timestamped setup stages (env and
# scene creation, first render, compile), a progress line every --log_freq steps (step, SPS, ETA, return, success,
# losses) instead of the tqdm bar, and each evaluation's result.   tail -f squint-lift-cube-dr-wrist-office3-log-<jobid>.out

# Wrist camera ONLY (single policy view, obs key "rgb").
# Options: wrist | third | wrist_third. Deploy with the same value.
export SQUINT_CAMERA_TYPE=wrist
# Office3 rig: wrist mount fitted to the office3 captures, office3 room (desk, wall, floor, D435 stand camera).
# Values live in WRIST_CAMERA_MOUNTS["office3"] and SCENES["office3"] (envs/base_random_env.py).
export SQUINT_WRIST_MOUNT=office3
export SQUINT_SCENE=office3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate squint

cd ~/projects/squint

date
nvidia-smi
echo "SQUINT_CAMERA_TYPE=${SQUINT_CAMERA_TYPE} SQUINT_WRIST_MOUNT=${SQUINT_WRIST_MOUNT} SQUINT_SCENE=${SQUINT_SCENE}"
echo "git: $(git log -1 --format='%h %s' 2>/dev/null) $(git status --porcelain 2>/dev/null | grep -q . && echo '(uncommitted changes)')"

# Table top: random color per env (any RGB)
# Shadows: angled light casts shadows, cameras use the slower "default" shader (the office3 colors are matched to it)
python -u train_squint.py \
    --env_id=SO101LiftCube-v1 \
    --exp_name=lift_cube_dr_wrist_office3_table_random_shadows_500k_log \
    --total_timesteps=500000 \
    --log_freq=10000 \
    --remove_overlay \
    --table_color=random \
    --shadows \
    --track \
    --wandb_entity=tsangb34-national-university-of-singapore-students-union \
    --wandb_project_name=cs6283 \
    --wandb_group=SQUINT
echo "train_squint.py exit code: $?"
date
