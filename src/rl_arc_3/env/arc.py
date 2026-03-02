import arc_agi
from arcengine import GameState

from rl_arc_3.base.env import EnvInterface, Observation


class ArcEnv(EnvInterface):
    def __init__(self, game: str = "ls20", render_mode: str = "terminal-fast"):
        arc = arc_agi.Arcade()
        self._env = arc.make(game, render_mode=render_mode)
        if self._env is None:
            raise ValueError(f"Failed to create environment for game: {game}")
        self.total_reward = 0.0

    def reset(self) -> Observation:
        obs = self._env.reset()
        self.total_reward = 0.0
        return Observation(frame=obs.frame, reward=0.0, terminated=False, info={})

    def step(self, action) -> Observation:
        obs = self._env.step(action)
        reward = 0.0
        if obs.levels_completed > self.total_reward:
            reward = obs.levels_completed - self.total_reward
            self.total_reward = obs.levels_completed
        else:
            reward = -0.001  # small negative reward to encourage progress

        terminated = obs.state in {GameState.WIN, GameState.GAME_OVER}

        return Observation(
            frame=obs.frame, reward=reward, terminated=terminated, info={}
        )
