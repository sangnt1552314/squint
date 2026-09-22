import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from mani_skill.utils import common

# ---------------------------  Wrappers --------------------------------------#


class MultiCameraObsWrapper(gym.ObservationWrapper):
    """Flattens ManiSkill observations into explicitly named per-camera RGB keys + state.

    Replaces FlattenRGBDObservationWrapper: cameras are selected by sensor name (never by
    dict ordering) and each camera keeps its own observation key instead of being
    concatenated along the channel dim.

    Args:
        camera_to_key: ordered {sim sensor name: obs key}, e.g.
            {"base_camera": "rgb"} or {"wrist_camera": "wrist_rgb", "third_camera": "third_rgb"}
    """

    def __init__(self, env, camera_to_key: dict):
        self.base_env = env.unwrapped
        super().__init__(env)
        self.camera_to_key = dict(camera_to_key)
        self.rgb_keys = tuple(self.camera_to_key.values())
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        sensor_data = observation.pop("sensor_data")
        observation.pop("sensor_param", None)

        ret = dict()
        for sensor_name, obs_key in self.camera_to_key.items():
            if sensor_name not in sensor_data:
                raise KeyError(
                    f"Policy camera '{sensor_name}' not found in sensor data "
                    f"(available: {sorted(sensor_data.keys())}). Check CAMERA_TYPE in "
                    f"envs/base_random_env.py and your real robot camera names."
                )
            ret[obs_key] = sensor_data[sensor_name]["rgb"]

        # flatten the rest of the data which should just be state data
        ret["state"] = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        return ret


class DownsampleObsWrapper(gym.ObservationWrapper):
    """Downsamples RGB observations from render_size to target_size using area interpolation.

    Each camera in `rgb_keys` is downsampled independently. Expects (B, H, W, C) format.
    """
    def __init__(self, env, target_size, rgb_keys=("rgb",)):
        super().__init__(env)
        self.target_size = target_size
        self.rgb_keys = tuple(rgb_keys)
        # Update observation space
        for key in self.rgb_keys:
            old_rgb_space = self.observation_space[key]
            C = old_rgb_space.shape[-1]
            self.observation_space[key] = gym.spaces.Box(
                low=0, high=255, shape=(target_size, target_size, C), dtype=old_rgb_space.dtype
            )

    def observation(self, obs):
        for key in self.rgb_keys:
            rgb = obs[key]  # (B, H, W, C) or (H, W, C)
            if rgb.shape[-2] == self.target_size:
                continue  # Already at target size

            # Handle batched and unbatched cases
            squeeze = rgb.dim() == 3
            if squeeze:
                rgb = rgb.unsqueeze(0)

            # (B, H, W, C) -> (B, C, H, W) for interpolate
            rgb = rgb.permute(0, 3, 1, 2)
            rgb = F.interpolate(rgb.float(), size=(self.target_size, self.target_size), mode='area').to(torch.uint8)
            # (B, C, H, W) -> (B, H, W, C)
            rgb = rgb.permute(0, 2, 3, 1)

            if squeeze:
                rgb = rgb.squeeze(0)

            obs[key] = rgb
        return obs



class ColorJitterWrapper(gym.ObservationWrapper):
    """Applies random color jitter to RGB observations for sim2real robustness.

    Expects input in (B, H, W, C) format.
    """
    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, rgb_keys=("rgb",)):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.rgb_keys = tuple(rgb_keys)

    def observation(self, obs):
        # Each camera is jittered independently (they are separate physical views)
        for key in self.rgb_keys:
            rgb = obs[key]  # (B, H, W, C) or (H, W, C) uint8

            # Handle batched and unbatched cases
            squeeze = rgb.dim() == 3
            if squeeze:
                rgb = rgb.unsqueeze(0)

            # (B, H, W, C) -> (B, C, H, W) for ColorJitter
            rgb = rgb.permute(0, 3, 1, 2)
            rgb = self.jitter(rgb.float() / 255.0)
            # (B, C, H, W) -> (B, H, W, C)
            rgb = rgb.permute(0, 2, 3, 1)

            # Back to uint8
            rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

            if squeeze:
                rgb = rgb.squeeze(0)

            obs[key] = rgb
        return obs


# ---------------------------  Extra Utils --------------------------------------#

def calc_buffer_memory(rgb_dim, state_dim, action_dim, max_length, rgb_dtype=np.uint8, store_next_obs=True):
    """Calculate memory required for buffer in GB and print it.

    Args:
        rgb_dim: Flattened dimension of all rgb observations summed over cameras
        state_dim: Dimension of state observation
        action_dim: Dimension of action space
        max_length: Maximum buffer length
        rgb_dtype: Data type for rgb storage 
        store_next_obs: Whether buffer stores next_obs separately (2x memory for obs)
    """
    obs_multiplier = 2 if store_next_obs else 1

    rgb_bytes = max_length * rgb_dim * np.dtype(rgb_dtype).itemsize * obs_multiplier
    state_bytes = max_length * state_dim * np.dtype(np.float32).itemsize * obs_multiplier
    act_bytes = max_length * action_dim * np.dtype(np.float32).itemsize
    other_bytes = max_length * np.dtype(np.float32).itemsize * 3

    # Total memory in GB
    total_gb = (rgb_bytes + state_bytes + act_bytes + other_bytes) / (1024**3)

    return total_gb





