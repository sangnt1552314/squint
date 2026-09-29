#!/bin/bash
# Deploy the wrist-only stack-cube policy trained for the OFFICE wrist mount (scripts/train/train_wrist_office_stack.sh).
# Usage: bash scripts/deploy/deploy_wrist_office_stack.sh [extra deploy.py args, e.g. --no-continuous_eval]

# Must match the setup the checkpoint was trained with (checked when loading the checkpoint).
# Real camera: WRIST_CAMERA_ID in deploy_utils/robot_config.py
export SQUINT_CAMERA_TYPE=wrist
export SQUINT_WRIST_MOUNT=office

cd "$(dirname "$0")/../.."

CHECKPOINT=runs/stack_cube_dr_wrist_office_table_random_shadows_1500k/ckpt.pt
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "SQUINT_CAMERA_TYPE=${SQUINT_CAMERA_TYPE} SQUINT_WRIST_MOUNT=${SQUINT_WRIST_MOUNT}"
echo "CHECKPOINT=${CHECKPOINT}"

# --record_dir: saves the preprocessed wrist camera image as one mp4 per episode
RECORD_DIR=recordings/stack_cube_dr_wrist_office_table_random_shadows_1500k/$(date +%Y%m%d_%H%M%S)
echo "RECORD_DIR=${RECORD_DIR}"

python -u deploy.py \
    --checkpoint="$CHECKPOINT" \
    --env_id=SO101StackCube-v1 \
    --policy_image_size=16 \
    --debug \
    --record_dir="$RECORD_DIR" \
    "$@"
