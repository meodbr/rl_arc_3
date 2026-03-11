import os
import logging

import numpy as np
import gymnasium as gym
import arc_agi
from arcengine import GameState,  GameAction

from rl_arc_3.base.env import BaseEnv, Envinfo, EnvSignature

logger = logging.getLogger(__name__)


class ArcEnv(BaseEnv):
    GRID_WIDTH: int = 64
    GRID_HEIGHT: int = 64

    def __init__(self, game: str = "ls20", render_mode: str = "terminal-fast"):
        arc = arc_agi.Arcade()

        arc_key = os.environ.get("ARC_API_KEY", "")
        assert arc_key != "", "Must set ARC_API_KEY env var."

        self._env = arc.make(game, render_mode=render_mode)
        if self._env is None:
            raise ValueError(f"Failed to create environment for game: {game}")
        self.total_reward = 0.0

        self._observation_space = gym.spaces.Box(low=0, high=15, shape=(64, 64), dtype=np.uint8)

        self._has_complex_action = any(act.is_complex() for act in self._env.action_space)
        self._simple_actions = [a for a in self._env.action_space if a.is_simple()]
        self._complex_actions = [a for a in self._env.action_space if a.is_complex()]

        assert len(self._complex_actions) < 2, "Error: env has more than 1 complex action"
        self._ordered_action_list = self._simple_actions + self._complex_actions


        if not self._has_complex_action:
            self._action_space = gym.spaces.Discrete(len(self._env.action_space))
        else:
            self._action_space = gym.spaces.Dict({
                "key": gym.spaces.Discrete(len(self._action_space)),
                "mouse": gym.spaces.Discrete(self.GRID_WIDTH * self.GRID_HEIGHT),
            })

    def reset(self) -> Envinfo:
        obs = self._env.reset()
        self.total_reward = 0.0
        return (obs.frame[-1], 0.0, False, {"total_reward": self.total_reward})

    def step(self, action) -> Envinfo:
        arc_action = self._to_arc_action(action)

        obs = self._env.step(arc_action)

        reward = 0.0
        if obs.levels_completed > self.total_reward:
            reward = obs.levels_completed - self.total_reward
            self.total_reward = obs.levels_completed
        else:
            reward = -0.001  # small negative reward to encourage progress

        terminated = obs.state in {GameState.WIN, GameState.GAME_OVER}

        logger.debug(obs)
        obs_frame = obs.frame[-1] if not terminated else None
        return (obs_frame, reward, terminated, {"total_reward": self.total_reward})
    
    def signature(self):
        return EnvSignature(
            observation_space=self._observation_space,
            action_space=self._action_space,
        )

    def _to_arc_action(self, action) -> GameAction:
        if not self._has_complex_action:
            return self._ordered_action_list[action]

        arc_action_id = self._ordered_action_list[action["key"]].value
        arc_action = GameAction.from_id(arc_action_id)

        if arc_action.is_complex():
            x = action["mouse"] % self.GRID_WIDTH
            y = action["mouse"] // self.GRID_WIDTH
            arc_action.set_data({"x": x, "y": y})
        else:
            if action["mouse"] != 0:
                logger.warning("Cursor coords are set (%d, %d) but selected action is not mouse action", action["mouse"] % self.GRID_WIDTH, action["mouse"] // self.GRID_WIDTH)

        return arc_action
        