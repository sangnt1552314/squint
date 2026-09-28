"""Robot configuration for real deployment. Edit the values below for your setup."""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras import Cv2Backends

from envs.base_random_env import CAMERA_TYPE, POLICY_CAMERA_NAMES

# ============================================================================
# CHANGE THESE: your hardware settings
# ============================================================================
ROBOT_PORT = "/dev/ttyACM0"        # your robot's serial port
ROBOT_ID = "home_follower"    # your calibration file name

# OpenCV camera IDs. On macOS these are plain integers (0, 1, 2, ...).
# On Linux they may be device paths such as "/dev/video0".
# Run `python deploy_utils/robot_config.py` to list what the machine sees.
WRIST_CAMERA_ID = 0
THIRD_CAMERA_ID = 1

CAMERA_FPS = 25
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# ============================================================================


def _opencv_camera(index_or_path):
    return OpenCVCameraConfig(
        index_or_path=index_or_path,
        fps=CAMERA_FPS,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        backend=Cv2Backends.AVFOUNDATION,
    )
    # RealSense alternative:
    # return RealSenseCameraConfig(
    #     serial_number_or_name="053645021390",
    #     fps=CAMERA_FPS, width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
    # )


def create_camera_configs() -> dict:
    """Build the LeRobot camera dict keyed by the *sim sensor names* in use.

    Sim2RealEnv looks real cameras up by the simulation sensor name, so the keys here
    must match POLICY_CAMERA_NAMES exactly (driven by CAMERA_TYPE in
    envs/base_random_env.py).

        CAMERA_TYPE = "wrist"       -> {"base_camera": wrist cam}
        CAMERA_TYPE = "third"       -> {"base_camera": third cam}
        CAMERA_TYPE = "wrist_third" -> {"wrist_camera": ..., "third_camera": ...}
    """
    if CAMERA_TYPE == "wrist":
        return {"base_camera": _opencv_camera(WRIST_CAMERA_ID)}
    if CAMERA_TYPE == "third":
        return {"base_camera": _opencv_camera(THIRD_CAMERA_ID)}
    if CAMERA_TYPE == "wrist_third":
        return {
            "wrist_camera": _opencv_camera(WRIST_CAMERA_ID),
            "third_camera": _opencv_camera(THIRD_CAMERA_ID),
        }
    raise ValueError(f"Unknown CAMERA_TYPE: {CAMERA_TYPE}")


def create_real_robot() -> Robot:
    """Create and configure a real robot with the cameras required by CAMERA_TYPE.

    Returns:
        Configured Robot instance
    """
    cameras = create_camera_configs()
    assert tuple(cameras.keys()) == tuple(POLICY_CAMERA_NAMES), (
        f"Real camera names {tuple(cameras.keys())} do not match the simulation policy "
        f"cameras {tuple(POLICY_CAMERA_NAMES)}."
    )

    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        use_degrees=True,
        cameras=cameras,
        id=ROBOT_ID,
        calibration_dir=Path(__file__).parent,  # CHANGE THIS: path to calibration file directory
    )

    return make_robot_from_config(robot_config)


if __name__ == "__main__":
    # Helper: probe which OpenCV camera indices are available on this machine
    import cv2

    print(f"CAMERA_TYPE={CAMERA_TYPE} -> expected camera names {tuple(POLICY_CAMERA_NAMES)}")
    print(f"Configured: WRIST_CAMERA_ID={WRIST_CAMERA_ID}, THIRD_CAMERA_ID={THIRD_CAMERA_ID}\n")
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            shape = frame.shape if ok else "read failed"
            print(f"  index {idx}: available ({shape})")
        cap.release()
