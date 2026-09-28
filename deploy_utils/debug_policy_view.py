"""
DEBUG ONLY: compare what the policy sees on the real robot vs in simulation.

Reads the real robot's current joint angles, puts the sim robot in exactly the same
configuration, and renders the sim wrist camera with the nominal camera parameters
(no domain randomization). Saves one image:

    row 1:  real preprocessed (128x128) | sim (128x128)       | 50/50 overlay
    row 2:  real policy input (16x16)   | sim policy input    | |real - sim| at 16x16

16x16 images are enlarged with nearest-neighbour so every policy pixel is visible.
MAE values are printed as a rough debugging aid only (backgrounds differ by design).

The real image goes through the exact deploy path (deploy.create_camera_preprocessor and
train_squint.DeployAgent.downsample_rgb); the sim image through the same downsample.

By default nothing moves: the bus is only read (torque stays as it is). Position the arm
by hand (torque off) or pass --start_pose to move the arm to the sim start pose first
(the robot MOVES; it is returned to rest afterwards).

Usage:
    SQUINT_CAMERA_TYPE=wrist python deploy_utils/debug_policy_view.py
    SQUINT_CAMERA_TYPE=wrist python deploy_utils/debug_policy_view.py --start_pose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import types

import cv2
import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401  (registers tasks)
import utils
from envs.base_random_env import CAMERA_TYPE, POLICY_CAMERAS, POLICY_CAMERA_NAMES, POLICY_RGB_KEYS, WRIST_MOUNT
from deploy import create_camera_preprocessor
from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot
from train_squint import DeployAgent


def to_cell(img: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (size, size), interpolation=interp)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="SO101LiftCube-v1")
    parser.add_argument("--image_size", type=int, default=128, help="sim/real render size (training --render_size)")
    parser.add_argument("--policy_image_size", type=int, default=16, help="policy input size (training --image_size)")
    parser.add_argument("--overlay", action="store_true", help="render sim with the black overlay (default: off, like --remove_overlay)")
    parser.add_argument("--start_pose", action="store_true", help="MOVE the real arm to the sim start pose before capturing")
    parser.add_argument("--out", default="debug_policy_view.png")
    parser.add_argument("--cell", type=int, default=256, help="display size of each cell")
    args = parser.parse_args()

    print(f"CAMERA_TYPE={CAMERA_TYPE}, policy cameras: {dict(POLICY_CAMERAS)}, SQUINT_WRIST_MOUNT={WRIST_MOUNT}")

    # --- Sim env: nominal camera, no randomization, no initial qpos noise ---
    sim_env = gym.make(
        args.env_id, obs_mode="rgb+segmentation", render_mode="sensors", num_envs=1,
        domain_randomization=False,
        sensor_configs=dict(width=args.image_size, height=args.image_size),
        domain_randomization_config=dict(apply_overlay=args.overlay, initial_qpos_noise_scale=0.0),
    )
    sim_env = utils.MultiCameraObsWrapper(sim_env, camera_to_key=POLICY_CAMERAS)
    sim_env.reset(seed=0)
    base = sim_env.unwrapped

    # --- Real robot: bus + cameras; torque untouched unless --start_pose ---
    robot = create_real_robot()
    agent = LeRobotRealAgent(robot)
    if args.start_pose:
        robot.connect()  # enables torque
        start_qpos = torch.tensor(base.agent.keyframes["start"].qpos, dtype=torch.float32)
        print("Moving real arm to start pose:", np.round(np.degrees(start_qpos.numpy()), 1))
        agent.reset(start_qpos)
        time.sleep(1.0)
    else:
        robot.bus.connect()
        for cam in robot.cameras.values():
            cam.connect()

    try:
        agent._cached_qpos = None
        real_qpos = agent.get_qpos()[0]
        print("real qpos (deg):", np.round(np.degrees(real_qpos.numpy()), 1))
        if args.start_pose:
            print("sim start qpos (deg):", np.round(np.degrees(base.agent.keyframes["start"].qpos), 1))

        # Put the sim robot in exactly the real configuration and move the wrist camera with it
        base.agent.robot.set_qpos(real_qpos.unsqueeze(0))
        if hasattr(base, "_update_wrist_camera_pose"):
            base._update_wrist_camera_pose()
        sim_obs = sim_env.observation(base.get_obs())

        # Real frame through the exact deploy preprocessing (center crop + cv2.resize to sim size)
        for cam in robot.cameras.values():
            for _ in range(10):  # let auto-exposure settle
                cam.async_read(timeout_ms=1000)
        agent.capture_sensor_data(list(POLICY_CAMERA_NAMES))
        real_data = create_camera_preprocessor(base)(agent.get_sensor_data(list(POLICY_CAMERA_NAMES)))

        downsampler = types.SimpleNamespace(target_image_size=args.policy_image_size)
        rows = []
        for sensor_name, key in zip(POLICY_CAMERA_NAMES, POLICY_RGB_KEYS):
            real = real_data[sensor_name]["rgb"]            # (1, H, W, 3) uint8 RGB
            sim = sim_obs[key].cpu()                         # (1, H, W, 3) uint8 RGB
            real_p = DeployAgent.downsample_rgb(downsampler, real)[0].numpy()
            sim_p = DeployAgent.downsample_rgb(downsampler, sim)[0].numpy()
            real, sim = real[0].numpy(), sim[0].numpy()

            mae_full = np.abs(real.astype(np.float32) - sim.astype(np.float32)).mean()
            mae_pol = np.abs(real_p.astype(np.float32) - sim_p.astype(np.float32)).mean()
            print(f"[{key}] MAE {args.image_size}x{args.image_size}: {mae_full:.1f}   "
                  f"MAE policy {args.policy_image_size}x{args.policy_image_size}: {mae_pol:.1f}   (0-255 scale)")

            c = args.cell
            overlay = (0.5 * real.astype(np.float32) + 0.5 * sim.astype(np.float32)).astype(np.uint8)
            diff = np.abs(real_p.astype(np.int16) - sim_p.astype(np.int16)).astype(np.uint8)
            rows.append(np.hstack([
                label(to_cell(real, c, False), f"{key} real {real.shape[0]}px"),
                label(to_cell(sim, c, False), f"{key} sim {sim.shape[0]}px"),
                label(to_cell(overlay, c, False), "overlay"),
            ]))
            rows.append(np.hstack([
                label(to_cell(real_p, c, True), f"real policy {args.policy_image_size}px"),
                label(to_cell(sim_p, c, True), f"sim policy {args.policy_image_size}px"),
                label(to_cell(diff, c, True), f"|diff| MAE {mae_pol:.0f}"),
            ]))

        cv2.imwrite(args.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        print(f"Saved {os.path.abspath(args.out)}")
    finally:
        if args.start_pose:
            print("Returning real arm to rest pose...")
            try:
                agent.reset(torch.tensor(base.agent.keyframes["rest"].qpos, dtype=torch.float32))
            except Exception as e:
                print(f"Warning: failed to return to rest: {e}")
            robot.disconnect()
        else:
            for cam in robot.cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
            try:
                robot.bus.disconnect(disable_torque=False)
            except Exception:
                pass
        sim_env.close()


if __name__ == "__main__":
    main()
