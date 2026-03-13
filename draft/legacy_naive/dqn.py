import logging
import random
import math

import torch
import torch.nn.functional as F

from rl_arc_3.base.env import Observation, Action, Transitions
from rl_arc_3.base.agent import (
    AgentConfig,
    AgentInterface,
    InferenceConfig,
    PolicyOutput,
)
from rl_arc_3.base.model import ModelFactory

from rl_arc_3.model.conv_basic import ConvBasicModule
from rl_arc_3.model.memory import TensorMemory, DequeMemory
from rl_arc_3.env.arc import ArcEnv

from .dqn_legacy import DQNModel

logger = logging.getLogger(__name__)


class DQNConfig(AgentConfig):
    gamma: float = 0.99
    lr: float = 1e-3
    eps_max: float = 0.9
    eps_min: float = 0.02
    eps_decay: float = 25000
    tau: float = 0.005
    device: str = "cpu"


class DQNInferenceConfig(InferenceConfig):
    weights: str = "target"


class DQNAgent(AgentInterface):
    """
    Class to wrap DQN training process
    """

    def __init__(
        self,
        config: DQNConfig,
        model_factory: ModelFactory,
        trainable: bool,
    ):
        self.device = config.device
        self.trainable = trainable
        self.model_factory = model_factory
        self.device = config.device
        self.config = config

        # Model instantiation
        self.model_kwargs = {
            "observation_space": config.observation_space,
            "action_space": config.action_space,
        }
        self.model = None
        if trainable:
            self.model = model_factory(**self.model_kwargs).to(self.device)
        self.target_model = model_factory(**self.model_kwargs).to(self.device).eval()

        logger.debug("DQNAgent instance with config: %s", config)

        self.target_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

    def __call__(self, observation: Observation) -> Action:
        return self.policy(
            observation=observation,
        ).selected_action

    @torch.no_grad
    def policy(
        self, observation: Observation, config: DQNInferenceConfig | None = None
    ):
        if config is None:
            config = DQNConfig()
        model = None
        match config.weights:
            case "online":
                if not self.trainable:
                    raise ValueError(
                        "online weights unavailable because agent is not trainable"
                    )
                model = self.model
            case "target":
                model = self.target_model
            case default:
                raise ValueError("Wrong weights for inference: %s", default)

        inputs = self.observation_to_tensor(observation)
        logits = model.forward(inputs)
        return PolicyOutput(
            selected_action=self.tensor_to_action(logits),
            logits=logits,
            info={},
        )

    def learn(self, batch):
        if not self.trainable:
            raise ValueError("Agent is not trainable")
        return super().learn(batch)

    def observation_to_tensor(self, observation: Observation) -> torch.Tensor:
        raise NotImplementedError

    def tensor_to_action(self, tensor: torch.Tensor) -> Action:
        raise NotImplementedError

    def compute_sample_batch(self, transitions: Transitions):
        # Sample memory
        transitions = self.memory.sample(batch_size)
        for tensor in transitions:
            if not tensor.device == self.device:
                tensor.to(self.device)
        state, action, next_state, reward, is_final = transitions
        # print(f"state shape: {state.shape}, dtype: {state.dtype}, device: {state.device}")

        # Compute: predicted = Q(s, a)
        predicted = self.model(state).gather(1, action)

        # Compute: expected = r + gamma * max_a(Q'(s',a))
        with torch.no_grad():
            next_state_reward = torch.zeros((batch_size, 1), device=self.device)
            next_state_reward[~is_final] = (
                self.target_model(next_state[~is_final]).max(1).values.unsqueeze(1)
            )
            expected = reward + self.gamma * next_state_reward

        return (predicted, expected)

    def train_iterations(self, n_iterations, batch_size=None) -> None:
        if not batch_size:
            batch_size = self.BATCH_SIZE

        if len(self.memory) < batch_size * 4:
            return

        self.model.train()
        for _ in range(n_iterations):
            self.train_step(batch_size)

    def train_step(self, batch_size=None):
        if not batch_size:
            batch_size = self.BATCH_SIZE

        if len(self.memory) < batch_size * 4:
            return

        self.model.train()
        self.optimizer.zero_grad()

        x_hat, x = self.compute_sample_batch(batch_size)

        loss = self.model.loss(x, x_hat)

        loss.backward()
        self.optimizer.step()

        self.update_target_model()

    def get_epsilon(self):
        return self.config.eps_min + (
            self.config.eps_max - self.config.eps_min
        ) * math.exp(-1 * (self.action_count / self.config.eps_decay))

    def select_action(self, observations: torch.Tensor, action_space_size: int) -> int:
        p = random.random()
        epsilon = self.get_epsilon()

        if p < epsilon:
            return torch.randint(0, action_space_size, (1,)).item()
        else:
            print(f"observations shape: {observations.shape}")
            print(f"observations dtype: {observations.dtype}")
            with torch.no_grad():
                logits = self.model(observations)
                print(f"logits: {logits}")
                return logits.argmax().item()

    def store_transition(self, transition: Tuple[torch.Tensor]):
        for elem in transition:
            if not isinstance(elem, torch.Tensor):
                ValueError("All elements of transition tuple must be tensors")

        ### Auto convertion code
        # transition = tuple(
        #     torch.as_tensor(elem, device=self.device) if not isinstance(elem, torch.Tensor)
        #     else elem.to(self.device)
        #     for elem in transition
        # )

        self.memory.push(transition)
        self.action_count += 1

    # Generic version of store_episode_statistics
    def store_episode_statistics(self, statistics: dict):
        """
        Store episode statistics in the statistics dictionary.
        If the key does not exist, it will be created.
        """
        for key, value in statistics.items():
            if key not in self.statistics:
                self.statistics[key] = []
            self.statistics[key].append(value)

    @torch.no_grad()
    def update_target_model(self):
        for target_param, policy_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data
            )

    def plot_statistics(self):
        """
        Plot statistics collected during training
        """
        if self.fig is None:
            self.fig = plt.figure(1, figsize=(14, 9))
        self.fig.clf()
        ax = self.fig.subplots(len(self.statistics) // 2 + 1, 2, sharex=True)

        x_axis = []
        sum = 0
        for x in self.statistics["duration"]:
            sum += x
            x_axis.append(sum)

        for i, (key, values) in enumerate(self.statistics.items()):
            ax[i // 2, i % 2].set_title(key)
            ax[i // 2, i % 2].plot(x_axis, np.array(values), label=key)

        self.fig.tight_layout()
        plt.pause(0.001)

        print("")
        print("Tprofiler statistics:")
        for key, times in self.tprof.items():
            values = np.array(times)
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f} seconds")


# Create an environment with terminal rendering
env = ArcEnv(game="ls20", render_mode="terminal-fast")

# Initialize model
model = DQNModel(
    model_class=ConvBasicModule,
    model_instantation_args={"size": 64, "channels": 16},
    memory=TensorMemory(
        capacity=1000, state_shape=(16, 64, 64), device=DQNModel.get_available_device()
    ),
)


def preprocess_frame(frame, device="cpu"):
    # Convert to tensor and add channel dimension (C, H, W)
    # 1 channel per color
    frame = torch.tensor(frame, dtype=torch.long, device=device)
    frame = F.one_hot(frame, num_classes=16).permute(2, 0, 1).float()  # (C, H, W)
    return frame


# Play the game
for episode in range(10000):
    obs = env.reset()
    done = False
    step_count = 0
    previous_frame = obs.frame[-1]
    frame = None
    total_reward = 0.0

    while not done:
        # Select an action (e-greedy)
        action_id = model.select_action(
            preprocess_frame(previous_frame, device=model.device), action_space_size=4
        )

        # Perform the action (rendering happens automatically)
        obs = env.step(action_id + 1)

        # Accumulate reward
        frame = obs.frame[-1]
        reward = obs.reward
        total_reward += reward
        done = obs.terminated

        # Store transition in memory
        transition = (
            preprocess_frame(previous_frame, device=model.memory.device),
            action_id,
            preprocess_frame(frame, device=model.memory.device),
            reward,
            done,
        )
        model.memory.push(transition)

        model.train_iterations(n_iterations=1, batch_size=32)

        previous_frame = frame
        step_count += 1

    print(
        f"Episode {episode + 1} finished in {step_count} steps with total reward {total_reward}"
    )

scorecard = env.get_scorecard()
if scorecard:
    print(f"Final Score: {scorecard.score}")
