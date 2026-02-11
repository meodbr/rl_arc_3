import gymnasium as gym
import ale_py

from rl_arc_3.base.env import EnvInterface, Observation

gym.register_envs(ale_py)


class AtariEnv(EnvInterface):
    def __init__(self, game: str = "pong", render_mode: str = "human"):
        self._env = gym.make(f"ALE/{game}-v5", render_mode=render_mode)
        self.total_reward = 0.0

    def reset(self) -> Observation:
        obs, info = self._env.reset()
        self.total_reward = 0.0
        return Observation(
            frame=obs,
            reward=0.0,
            terminated=False,
            info=info,
        )

    def step(self, action) -> Observation:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.total_reward += reward
        done = terminated or truncated

        return Observation(
            frame=obs,
            reward=reward,
            terminated=done,
            info=info,
        )
