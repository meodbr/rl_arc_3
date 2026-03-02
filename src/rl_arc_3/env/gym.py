import gymnasium as gym
import ale_py

from rl_arc_3.base.env import BaseEnv, EnvSignature, Envinfo

gym.register_envs(ale_py)


class AtariEnv(BaseEnv):
    def __init__(self, game: str = "pong", render_mode: str = "human"):
        self._env = gym.make(f"ALE/{game}-v5", render_mode=render_mode)
        self.total_reward = 0.0

    def reset(self) -> Envinfo:
        obs, info = self._env.reset()
        self.total_reward = 0.0
        return (obs, 0.0, False, info)

    def step(self, action) -> Envinfo:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.total_reward += reward
        done = terminated or truncated
        return (obs, reward, done, info)
    
    def env_signature(self) -> EnvSignature:
        return EnvSignature(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
        )
