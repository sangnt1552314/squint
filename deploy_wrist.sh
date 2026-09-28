#!/bin/bash
# Deploy the wrist-only lift-cube policy on the real SO101.
# Usage: bash deploy_wrist.sh [extra deploy.py args, e.g. --debug --record_dir=recordings/lift_wrist]

# Must match the camera setup the checkpoint was trained with (train_wrist.sh -> wrist).
# Real camera: WRIST_CAMERA_ID in deploy_utils/robot_config.py
export SQUINT_CAMERA_TYPE=wrist
# Original SQUINT/WowRobo wrist mount (camera sees the gripper jaws), as this checkpoint was trained
export SQUINT_WRIST_MOUNT=default

cd "$(dirname "$0")"

CHECKPOINT=runs/lift_cube_dr_wrist_1500k/ckpt.pt
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "SQUINT_CAMERA_TYPE=${SQUINT_CAMERA_TYPE} SQUINT_WRIST_MOUNT=${SQUINT_WRIST_MOUNT}"
echo "CHECKPOINT=${CHECKPOINT}"

# --debug: live window with real | sim | overlay of the wrist camera
# --record_dir: saves the preprocessed wrist camera image (the policy's input before the
#   16x16 downsample) as one mp4 per episode
RECORD_DIR=recordings/lift_cube_dr_wrist_1500k/$(date +%Y%m%d_%H%M%S)
echo "RECORD_DIR=${RECORD_DIR}"

python -u deploy.py \
    --checkpoint="$CHECKPOINT" \
    --env_id=SO101LiftCube-v1 \
    --policy_image_size=16 \
    --debug \
    --record_dir="$RECORD_DIR" \
    "$@"
