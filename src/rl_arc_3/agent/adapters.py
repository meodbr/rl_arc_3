from typing import Any, Tuple, Iterable
import logging

import torch
import torch.nn.functional as F
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box, Space

from rl_arc_3.base.env import EnvSignature
from rl_arc_3.base.model import ModelSignature
from rl_arc_3.base.model_adapter import ModelAdapter

from rl_arc_3.utils.utils import unwrap_if_single
from rl_arc_3.utils.constants import ENV_OBS_CHANNEL_DIM

logger = logging.getLogger(__name__)


class ArcStyleModelAdapter(ModelAdapter):
    """
    Adapter for "arc style" environments:
        - With Dict action spaces containing "key" and "mouse" Discrete subspaces.
        - Or with Discrete action spaces, i.e. no mouse (ex: Atari envs)
    """

    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        env_act, env_obs = self.validate_env_spaces(env_signature)

        self._is_action_env_discrete = isinstance(env_act, Discrete)

        self._env_obs_is_channel_dim_compressed = len(env_obs.shape) == 2
        self._env_obs_dtype = env_obs.dtype
        logger.info("Env obs shape=%s dtype=%s", env_obs.shape, env_obs.dtype)

        self._env_obs_high = env_obs.high.max()
        self._env_obs_low = env_obs.low.min()

        self._env_obs_is_integer = np.issubdtype(env_obs.dtype, np.integer)
        self._env_obs_is_binary = (
            self._env_obs_low == 0 and self._env_obs_high == 1
            if self._env_obs_is_integer
            else False
        )
        self._env_obs_should_rescale = (
            self._env_obs_is_integer and not self._env_obs_is_channel_dim_compressed
        )
        self._env_obs_rescale_factor = (
            1.0 / (self._env_obs_high - self._env_obs_low)
            if self._env_obs_should_rescale
            else 1.0
        )

        self._env_obs_is_compression_supported = False

        logger.info(
            "Env obs shape=%s dtype=%s, range=(%s, %s)",
            env_obs.shape,
            env_obs.dtype,
            self._env_obs_low,
            self._env_obs_high,
        )

        if self._env_obs_is_channel_dim_compressed:
            low = self._env_obs_low
            high = self._env_obs_high
            assert (
                low == 0
            ), f"Env obs space should have 0 as lowest pixel value, got {low}"
            self._env_obs_num_channels = high - low
            self._env_obs_is_compression_supported = True
        else:
            self._env_obs_num_channels = env_obs.shape[ENV_OBS_CHANNEL_DIM]
            self._env_obs_is_compression_supported = self._env_obs_is_binary

        if self._env_obs_num_channels > 32:
            logger.warning(
                "Detected number of input channels: %d, seems too large",
                self._env_obs_num_channels,
            )

        super().__init__(env_signature, model_signature)

    def validate_env_spaces(self, env_signature: EnvSignature) -> Tuple[Space, Space]:
        env_act = env_signature.action_space
        env_obs = env_signature.observation_space

        # Observation space
        if not isinstance(env_obs, Box):
            raise RuntimeError(
                f"Only Box observation spaces are supported, not {type(env_obs)}"
            )

        if len(env_obs.shape) not in [2, 3]:
            raise RuntimeError(
                f"{env_obs.shape} not in supported obs shapes : (H,W) or (H,W,C)"
            )

        # Action space
        if not isinstance(env_act, Dict) and not isinstance(env_act, Discrete):
            raise RuntimeError(
                f"Only Dict and Discrete action spaces are supported, not {type(env_act)}"
            )

        if isinstance(env_act, Dict):
            for name, subspace in env_act.spaces.items():
                if not isinstance(subspace, Discrete):
                    raise RuntimeError(
                        f"Only Discrete action subspaces are supported, not {type(subspace)}"
                    )
                if name not in ["key", "mouse"]:
                    raise RuntimeError(f"Unsupported action subspace name: {name}")

        return env_act, env_obs

    def compress_obs(self, t: torch.Tensor, batched: bool = False) -> np.ndarray:
        if not self._env_obs_is_compression_supported:
            return t.numpy()
        assert t.dim() == (4 if batched else 3), "B={}, shape={}, shape should be (B, C, H, W) if batched else (C, H, W)".format(batched, t.shape())
        return torch.argmax(t, dim=1 if batched else 0, dtype=torch.uint8).numpy()

    def uncompress_obs(self, t: np.ndarray, batched: bool = False, device: str = None) -> torch.Tensor:
        if not self._env_obs_is_compression_supported:
            return torch.tensor(t, device=device).float()

        assert t.dim() == (3 if batched else 2), "B={}, shape={}, shape should be (B, H, W) if batched else (H, W)".format(batched, t.shape())
        if batched:
            return F.one_hot(t, num_classes=self._env_obs_num_channels).permute(0, 3, 1, 2).to(device).float() # (B, C, H, W)
        else:
            return F.one_hot(t, num_classes=self._env_obs_num_channels).permute(2, 0, 1).to(device).float() # (C, H, W)
    
    def read_obs(self, obs: np.ndarray, device: str = None):
        assert ENV_OBS_CHANNEL_DIM == 2, "Channel dim not supported: {}".format(ENV_OBS_CHANNEL_DIM)

        if not self._env_obs_is_channel_dim_compressed:
            res = torch.tensor(obs.transpose(2, 0, 1), device=device, dtype=torch.float)
            if self._env_obs_should_rescale:
                res = res * self._env_obs_rescale_factor
            return res
        
        res = torch.tensor(obs, device=device)
        res = F.one_hot(res, num_classes=self._env_obs_num_channels).permute(2, 0, 1).float() # (C, H, W)
        return res

    @staticmethod
    def reorder_channel_dim(shape: Iterable[int]) -> Tuple:
        chan_dim = ENV_OBS_CHANNEL_DIM
        s = list(shape)
        return tuple([s[chan_dim]] + s[:chan_dim] + s[chan_dim+1:])


    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        raise NotImplementedError

    def observation_to_tensor(
        self, obs: Any, compressed: bool = False, device: str = None
    ) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    def tensor_to_action(self, array: torch.Tensor) -> Any:
        raise NotImplementedError


def get_model_adapter(
    name: str,
    env_signature: EnvSignature,
    model_signature: ModelSignature | None = None,
) -> ModelAdapter:
    if name == "full":
        return FullModelAdapter(env_signature, model_signature)
    elif name == "keyboard_only":
        return KeyboardOnlyModelAdapter(env_signature, model_signature)
    else:
        raise ValueError(f"Unknown model adapter name: {name}")


class KeyboardOnlyModelAdapter(ArcStyleModelAdapter):
    """
    This adapter does not use the "mouse" subspace (always sets it to 0) and only outputs actions for the "key" subspace.
    """

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=ArcStyleModelAdapter.reorder_channel_dim(env_signature.observation_space.shape),
                output_shape=[env_signature.action_space.n],
            )
        else:
            return ModelSignature(
                input_shape=ArcStyleModelAdapter.reorder_channel_dim(env_signature.observation_space.shape),
                output_shape=[env_signature.action_space.spaces["key"].n],
            )

    def observation_to_tensor(
        self, obs: np.ndarray, device: str = None
    ) -> torch.Tensor:
        return self.read_obs(obs, device=device)

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        key_action = x.tolist()

        if self._is_action_env_discrete:
            return unwrap_if_single(key_action)

        actions = [
            {
                "key": k_a,
                "mouse": 0,  # Mouse action not used in this adapter
            }
            for k_a in key_action
        ]

        return unwrap_if_single(actions)


class FullModelAdapter(ArcStyleModelAdapter):
    """
    This adapter uses the mouse action by splitting the mouse action into 1 action for each pixel
    """

    def __init__(
        self, env_signature: EnvSignature, model_signature: ModelSignature | None = None
    ):
        super().__init__(env_signature, model_signature)

        if not self._is_action_env_discrete:
            self.key_n = self.env_act.spaces["key"].n
            self.mouse_n = self.env_act.spaces["mouse"].n
            self.mouse_action_id = self.env_act.spaces["key"].n - 1

    @staticmethod
    def compute_model_signature(env_signature: EnvSignature) -> ModelSignature:
        if isinstance(env_signature.action_space, Discrete):
            return ModelSignature(
                input_shape=ArcStyleModelAdapter.reorder_channel_dim(env_signature.observation_space.shape),
                output_shape=[env_signature.action_space.n],
            )

        lenghts = {}
        for name, subspace in env_signature.action_space.spaces.items():
            lenghts[name] = subspace.n

        return ModelSignature(
            input_shape=ArcStyleModelAdapter.reorder_channel_dim(env_signature.observation_space.shape),
            output_shape=[
                lenghts["key"] - 1 + lenghts["mouse"]
            ],  # -1 because mouse action is represented as 1 action for each pixel
        )

    def observation_to_tensor(
        self, obs: np.ndarray, device: str = None
    ) -> torch.Tensor:
        return self.read_obs(obs, device=device)

    def tensor_to_action(self, x: torch.Tensor) -> Any:
        action_list = x.tolist()

        if self._is_action_env_discrete:
            return unwrap_if_single(action_list)

        num_key_actions = self.key_n - 1  # Exclude the "mouse" action
        key_action = [
            a if a < num_key_actions else self.mouse_action_id for a in action_list
        ]
        mouse_action = [
            a - num_key_actions if a >= self.key_n else 0 for a in action_list
        ]

        actions = [
            {
                "key": k_a,
                "mouse": m_a,
            }
            for k_a, m_a in zip(key_action, mouse_action)
        ]

        return unwrap_if_single(actions)
