"""
Live camera tuning for sim2real alignment (SO101).

Shows one "Real | Sim | Blended" row per policy camera (wrist and/or third-view,
driven by CAMERA_TYPE in envs/base_random_env.py). Trackbars adjust the camera
currently selected for tuning; press 'c' to switch between wrist and third.

Wrist trackbars: pose relative to gripper_link (x, y, z, roll, pitch, yaw) + FOV.
Third trackbars: world eye position + look-at target + FOV.

Keys: c=switch camera, p=print params, r=rest pose, s=start pose, f=apply FOV, q=quit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import signal
import atexit
import argparse

import cv2
import numpy as np
import torch
import gymnasium as gym
import sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose

import utils
from envs.base_random_env import CAMERA_TYPE, POLICY_CAMERAS, POLICY_CAMERA_NAMES, WRIST_MOUNT

from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot

import envs


# Which logical camera each sim sensor name corresponds to, per CAMERA_TYPE
_CAMERA_ROLES = {
    "wrist": {"base_camera": "wrist"},
    "third": {"base_camera": "third"},
    "wrist_third": {"wrist_camera": "wrist", "third_camera": "third"},
}[CAMERA_TYPE]


class LiveCameraTuner:
    def __init__(self, env_id: str, sim_width: int = 480, sim_height: int = 480, camera: str = None):
        self.env_id = env_id
        self.sim_width = sim_width
        self.sim_height = sim_height

        self.sensor_names = list(POLICY_CAMERA_NAMES)
        self.roles = _CAMERA_ROLES
        available_roles = list(dict.fromkeys(self.roles.values()))
        self.tuned_role = camera if camera in available_roles else available_roles[0]

        # Per-role camera parameters (overwritten by sim extraction)
        # wrist: pose relative to gripper_link
        self.wrist = dict(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, fov=71.0)
        # third: world-frame eye + look-at target
        self.third = dict(ex=0.6, ey=0.3, ez=0.3, tx=0.3, ty=0.0, tz=0.05, fov=60.0)

        self._last_fov = {}
        self._fov_pending = False

        # Trackbar scaling
        self.pos_scale = 1000  # mm

        self.sim_env = None
        self.real_robot = None
        self.real_agent = None

        self._create_sim_env()
        self._setup_real_robot()
        self._move_real_to_sim_pose()
        self._setup_exit()
        self._setup_ui()

    # --- Sim environment ---

    @property
    def params(self):
        """Parameter dict of the camera currently being tuned."""
        return self.wrist if self.tuned_role == "wrist" else self.third

    def _sensor_name_for(self, role):
        for name, r in self.roles.items():
            if r == role:
                return name
        return None

    def _create_sim_env(self, preserve_fov=False):
        desired = {r: (self.wrist if r == "wrist" else self.third)["fov"]
                   for r in set(self.roles.values())} if preserve_fov else None
        if self.sim_env is not None:
            self.sim_env.close()

        sensor_configs = {"width": self.sim_width, "height": self.sim_height}
        if desired is not None:
            # Per-camera FOV override (ManiSkill applies name-keyed entries per sensor)
            for name, role in self.roles.items():
                sensor_configs[name] = {"fov": np.radians(desired[role])}

        self.sim_env = gym.make(
            self.env_id,
            obs_mode="rgb+segmentation",
            render_mode="sensors",
            num_envs=1,
            domain_randomization=False,
            domain_randomization_config={"initial_qpos_noise_scale": 0.0},
            sensor_configs=sensor_configs,
        )
        self.sim_env = utils.MultiCameraObsWrapper(self.sim_env, camera_to_key=POLICY_CAMERAS)
        self.sim_env.reset(seed=0)
        self._extract_camera_params()

        if desired is not None:
            self.wrist["fov"] = desired.get("wrist", self.wrist["fov"])
            self.third["fov"] = desired.get("third", self.third["fov"])
        self._last_fov = {"wrist": self.wrist["fov"], "third": self.third["fov"]}

    def _extract_camera_params(self):
        """Extract camera params from the sim environment class constants."""
        env = self.sim_env.unwrapped

        if hasattr(env, "WRIST_CAMERA_BASE_POS"):
            pos = env.WRIST_CAMERA_BASE_POS
            rot = env.WRIST_CAMERA_BASE_ROT_RAD
            self.wrist.update(
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                roll=float(np.degrees(rot[0])),
                pitch=float(np.degrees(rot[1])),
                yaw=float(np.degrees(rot[2])),
                fov=float(np.degrees(env.WRIST_CAMERA_FOV)),
            )

        if hasattr(env, "DEFAULT_CAMERA_POS"):
            eye = env.base_camera_settings["pos"] if hasattr(env, "base_camera_settings") else env.DEFAULT_CAMERA_POS
            tgt = env.base_camera_settings["target"] if hasattr(env, "base_camera_settings") else env.DEFAULT_CAMERA_TARGET
            self.third.update(
                ex=float(eye[0]), ey=float(eye[1]), ez=float(eye[2]),
                tx=float(tgt[0]), ty=float(tgt[1]), tz=float(tgt[2]),
                fov=float(np.degrees(env.DEFAULT_CAMERA_FOV)),
            )

    # --- Real robot ---

    def _setup_real_robot(self):
        self.real_robot = create_real_robot()
        self.real_robot.connect()
        self.real_agent = LeRobotRealAgent(self.real_robot)

    def _move_real_to_sim_pose(self):
        if self.real_agent is None or self.sim_env is None:
            return
        qpos = self.sim_env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.squeeze()
        self.real_agent.reset(qpos)

    # --- Camera update ---

    def _update_cameras(self):
        env = self.sim_env.unwrapped

        if "wrist" in self.roles.values() and hasattr(env, "wrist_camera_mount"):
            w = self.wrist
            gripper_pose = env.agent.robot.links_map["gripper_link"].pose
            p_t = torch.tensor([[w["x"], w["y"], w["z"]]], dtype=torch.float32, device=env.device)
            r, p, y = np.radians(w["roll"]), np.radians(w["pitch"]), np.radians(w["yaw"])
            q = np.array(qmult(euler2quat(0, p, y, axes="rxyz"), euler2quat(r, 0, 0, axes="rxyz")), dtype=np.float32)
            q_t = torch.from_numpy(q).unsqueeze(0).to(device=env.device)
            env.wrist_camera_mount.set_pose(gripper_pose * Pose.create_from_pq(p=p_t, q=q_t))

        if "third" in self.roles.values() and hasattr(env, "camera_mount"):
            t = self.third
            pose = sapien_utils.look_at(eye=[t["ex"], t["ey"], t["ez"]],
                                        target=[t["tx"], t["ty"], t["tz"]])
            env.camera_mount.set_pose(Pose.create(pose.raw_pose.squeeze().unsqueeze(0)))

        if env.gpu_sim_enabled:
            env.scene._gpu_apply_all()

    # --- Image capture ---

    def _get_real_images(self):
        """{sensor_name: BGR image} for every policy camera."""
        self.real_agent.capture_sensor_data(self.sensor_names)
        obs = self.real_agent.get_sensor_data(self.sensor_names)
        images = {}
        for name in self.sensor_names:
            if name not in obs or "rgb" not in obs[name]:
                continue
            rgb = obs[name]["rgb"]
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]

            # Center-crop to square
            h, w = rgb.shape[:2]
            if h != w:
                s = min(h, w)
                c = (max(h, w) - s) // 2
                rgb = rgb[c:c + s, :, :] if h > w else rgb[:, c:c + s, :]

            rgb = cv2.resize(rgb, (self.sim_width, self.sim_height))
            images[name] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return images

    def _get_sim_images(self):
        """{sensor_name: BGR image} for every policy camera."""
        obs = self.sim_env.unwrapped.get_obs()
        images = {}
        for name in self.sensor_names:
            cam_data = obs.get("sensor_data", {}).get(name)
            if cam_data is None or "rgb" not in cam_data:
                continue
            rgb = cam_data["rgb"][0].cpu().numpy()
            images[name] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return images

    def _make_comparison(self, real_imgs, sim_imgs):
        """One 'Real | Sim | Blended' row per policy camera, stacked vertically."""
        rows = []
        font = cv2.FONT_HERSHEY_SIMPLEX

        for name in self.sensor_names:
            real = real_imgs.get(name)
            sim = sim_imgs.get(name)
            if real is None or sim is None:
                continue
            h, w = real.shape[:2]
            sim_r = cv2.resize(sim, (w, h))
            blended = cv2.addWeighted(real, 0.5, sim_r, 0.5, 0)
            row = np.hstack([real, sim_r, blended])

            role = self.roles[name]
            tuning = " [TUNING]" if role == self.tuned_role else ""
            for text, x_off in [(f"{name} Real{tuning}", 10), ("Sim", w + 10), ("Blended", 2 * w + 10)]:
                cv2.putText(row, text, (x_off, 40), font, 1.0, (0, 0, 0), 5)
                cv2.putText(row, text, (x_off, 40), font, 1.0, (255, 255, 255), 2)
            rows.append(row)

        if not rows:
            return None
        comp = np.vstack(rows)

        cv2.putText(comp, self._param_text(), (10, comp.shape[0] - 15), font, 0.7, (0, 0, 0), 3)
        cv2.putText(comp, self._param_text(), (10, comp.shape[0] - 15), font, 0.7, (255, 255, 255), 2)
        return comp

    def _param_text(self):
        if self.tuned_role == "wrist":
            w = self.wrist
            return (f"wrist pos=[{w['x']:.3f},{w['y']:.3f},{w['z']:.3f}] "
                    f"rot=[{w['roll']:.0f},{w['pitch']:.0f},{w['yaw']:.0f}] fov={w['fov']:.0f}")
        t = self.third
        return (f"third eye=[{t['ex']:.3f},{t['ey']:.3f},{t['ez']:.3f}] "
                f"target=[{t['tx']:.3f},{t['ty']:.3f},{t['tz']:.3f}] fov={t['fov']:.0f}")

    # --- UI ---

    def _setter(self, store, key, offset=0.0, scale=1.0):
        def fn(v):
            store[key] = v / scale - offset
        return fn

    def _setup_ui(self):
        self.win = "Live Camera Tuner | c:switch p:print r:rest s:start f:FOV q:quit"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        self._build_trackbars()

    def _build_trackbars(self):
        """(Re)create trackbars for the camera currently being tuned."""
        cv2.destroyWindow(self.win)
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        if self.tuned_role == "wrist":
            w = self.wrist
            # +-150mm around the gripper_link origin so side-mounted brackets are reachable
            for label, key in [("X (mm)", "x"), ("Y (mm)", "y"), ("Z (mm)", "z")]:
                cv2.createTrackbar(label, self.win, int((w[key] + 0.15) * self.pos_scale), 300,
                                   self._setter(w, key, 0.15, self.pos_scale))
            cv2.createTrackbar("Roll", self.win, int(w["roll"] + 180), 360, self._setter(w, "roll", 180))
            cv2.createTrackbar("Pitch", self.win, int(w["pitch"] + 180), 360, self._setter(w, "pitch", 180))
            cv2.createTrackbar("Yaw", self.win, int(w["yaw"] + 180), 360, self._setter(w, "yaw", 180))
        else:
            t = self.third
            # eye/target in mm over a +-1m workspace
            for label, key in [("Eye X (mm)", "ex"), ("Eye Y (mm)", "ey"), ("Eye Z (mm)", "ez"),
                               ("Tgt X (mm)", "tx"), ("Tgt Y (mm)", "ty"), ("Tgt Z (mm)", "tz")]:
                cv2.createTrackbar(label, self.win, int((t[key] + 1.0) * self.pos_scale), 2000,
                                   self._setter(t, key, 1.0, self.pos_scale))

        cv2.createTrackbar("FOV", self.win, int(self.params["fov"]), 120, self._on_fov)

    def _on_fov(self, val):
        new = max(10, val)
        if new != self.params["fov"]:
            self.params["fov"] = new
            self._fov_pending = True

    def _setup_exit(self):
        def cleanup(sig=None, frame=None):
            try:
                if self.real_agent and self.sim_env:
                    self.real_agent.reset(self.sim_env.unwrapped.agent.keyframes["rest"].qpos)
            except Exception:
                pass
            try:
                self.real_robot and self.real_robot.disconnect()
            except Exception:
                pass
            try:
                self.sim_env and self.sim_env.close()
            except Exception:
                pass
            if sig is not None:
                sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        atexit.register(cleanup)
        self._cleanup = cleanup

    def print_params(self):
        print(f"\n{'='*70}")
        if self.tuned_role == "wrist":
            w = self.wrist
            print(f"Wrist camera preset (paste into WRIST_CAMERA_MOUNTS in envs/base_random_env.py, "
                  f"current SQUINT_WRIST_MOUNT={WRIST_MOUNT}):")
            print(f'  "{WRIST_MOUNT}": dict(pos=({w["x"]:.4f}, {w["y"]:.4f}, {w["z"]:.4f}), '
                  f'rot_deg=({w["roll"]:.1f}, {w["pitch"]:.1f}, {w["yaw"]:.1f}), fov_deg={w["fov"]:.1f}),')
        else:
            t = self.third
            print("Third camera params for ThirdCameraEnv (envs/base_random_env.py):")
            print(f"  DEFAULT_CAMERA_POS = [{t['ex']:.4f}, {t['ey']:.4f}, {t['ez']:.4f}]")
            print(f"  DEFAULT_CAMERA_TARGET = [{t['tx']:.4f}, {t['ty']:.4f}, {t['tz']:.4f}]")
            print(f"  DEFAULT_CAMERA_FOV = np.deg2rad({t['fov']:.1f})")
        print(f"{'='*70}\n")

    def run(self):
        roles = list(dict.fromkeys(self.roles.values()))
        print(f"\nCAMERA_TYPE={CAMERA_TYPE}, policy cameras: {dict(POLICY_CAMERAS)}, SQUINT_WRIST_MOUNT={WRIST_MOUNT}")
        print("\nControls:")
        print(f"  c  - Switch tuned camera (available: {roles})")
        print("  p  - Print current camera parameters")
        print("  r  - Move sim+real to rest pose")
        print("  s  - Move sim+real to start pose")
        print("  f  - Apply pending FOV change")
        print("  q  - Quit")
        print("  Trackbars - Adjust the tuned camera\n")

        while True:
            self._update_cameras()
            comp = self._make_comparison(self._get_real_images(), self._get_sim_images())

            if comp is not None:
                pending = self.params["fov"] != self._last_fov.get(self.tuned_role)
                if pending:
                    txt = f"FOV: {self._last_fov[self.tuned_role]:.0f}->{self.params['fov']:.0f} (press 'f')"
                    cv2.putText(comp, txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(comp, txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow(self.win, comp)
            else:
                err = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
                cv2.putText(err, "Waiting for camera...", (700, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.imshow(self.win, err)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                if len(roles) > 1:
                    self.tuned_role = roles[(roles.index(self.tuned_role) + 1) % len(roles)]
                    self._build_trackbars()
                    print(f"Now tuning: {self.tuned_role} camera")
            elif key == ord("p"):
                self.print_params()
            elif key == ord("r"):
                try:
                    rest_qpos = self.sim_env.unwrapped.agent.keyframes["rest"].qpos
                    qpos = rest_qpos if isinstance(rest_qpos, torch.Tensor) else torch.tensor(rest_qpos, dtype=torch.float32)
                    if qpos.dim() == 1:
                        qpos = qpos.unsqueeze(0)
                    env = self.sim_env.unwrapped
                    env.agent.robot.set_qpos(qpos)
                    if env.gpu_sim_enabled:
                        env.scene._gpu_apply_all()
                    self.real_agent.reset(rest_qpos)
                    print("Moved sim+real to rest pose")
                except Exception as e:
                    print(f"Rest pose error: {e}")
            elif key == ord("s"):
                try:
                    self.sim_env.reset(seed=0)
                    self._move_real_to_sim_pose()
                    print("Moved sim+real to start pose")
                except Exception as e:
                    print(f"Start pose error: {e}")
            elif key == ord("f") and self._fov_pending:
                self._create_sim_env(preserve_fov=True)
                self._build_trackbars()
                self._fov_pending = False

        cv2.destroyAllWindows()
        self._cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live camera tuning (SO101)")
    parser.add_argument("--env-id", default="SO101ReachCube-v1", help="Sim environment ID")
    parser.add_argument("--camera", default=None, choices=["wrist", "third"],
                        help="Which camera the trackbars start on (press 'c' to switch)")
    parser.add_argument("--sim-width", type=int, default=480)
    parser.add_argument("--sim-height", type=int, default=480)
    args = parser.parse_args()
    LiveCameraTuner(args.env_id, args.sim_width, args.sim_height, args.camera).run()
