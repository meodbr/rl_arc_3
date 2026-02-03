from typing import Iterable, Any
from dataclasses import dataclass, field

from gymnasium.spaces import Space

from rl_arc_3.env.interface import Observation, Action, Transitions

@dataclass
class PolicyOutput:
    selected_action: Action
    logits: Any = None
    info: dict = field(default_factory=dict)

@dataclass
class InferenceConfig:
    return_logits: bool = True
    weights: str = "target"

@dataclass
class AgentConfig:
    observation_space: Space
    action_space: Space


class AgentInterface:
    def __call__(
        self,
        observation: Observation,
    ) -> Action:
        """Default inference: returns deterministic action"""
        raise NotImplementedError

    def policy(
        self,
        observation: Observation,
        config: InferenceConfig | None = None,
    ) -> PolicyOutput:
        raise NotImplementedError
    
    def learn(
        self,
        batch: Transitions,
    ) -> dict:
        raise NotImplementedError

    def build_transitions(
        self,
        observations: Iterable[Observation],
        model_outputs: Iterable[PolicyOutput],
    ) -> Transitions:
        raise NotImplementedError
    
    def state_dict(
        self
    ) -> dict:
        raise NotImplementedError
    
    def load_state_dict(
        self,
        state: dict
    ) -> None:
        raise NotImplementedError