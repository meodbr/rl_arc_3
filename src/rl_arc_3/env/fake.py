import gymnasium as gym
import ale_py
import numpy as np

from rl_arc_3.base.env import BaseEnv, EnvSignature, Envinfo

gym.register_envs(ale_py)


class FakeEnv(BaseEnv):
    def __init__(self, game: str = "fake_game", render_mode: str = "human"):
        self._env = None
        self._action_space = gym.spaces.Dict({
            "key": gym.spaces.Discrete(5),
            "mouse": gym.spaces.Discrete(64 * 64),
        })
        self._observation_space = gym.spaces.Box(low=0, high=16, shape=(64, 64), dtype=np.uint8)
        self.game = game
        self.render_mode = render_mode

    def reset(self) -> Envinfo:
        obs = self._observation_space.sample()
        self.total_reward = 0.0
        return (obs, 0.0, False, {"total_reward": self.total_reward})

    def step(self, action) -> Envinfo:
        obs = self._observation_space.sample()
        if action["key"] == 0:
            reward = 1.0
        elif action["key"] == 4 and action["mouse"] == 42:
            reward = 10.0
        else:
            reward = 0.0
        done = np.random.rand() < 0.01
        self.total_reward += reward
        return (obs, reward, done, {"total_reward": self.total_reward})
    
    def signature(self) -> EnvSignature:
        return EnvSignature(
            observation_space=self._observation_space,
            action_space=self._action_space,
        )
