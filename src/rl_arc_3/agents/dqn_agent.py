import torch
import torch.nn.functional as F

from rl_arc_3.models.dqn import DQNModel, ConvBasicModule
from rl_arc_3.models.memory import TensorMemory, DequeMemory
from rl_arc_3.env.arc import ArcEnv

# Create an environment with terminal rendering
env = ArcEnv(game="ls20", render_mode="terminal-fast")

# Initialize model
model = DQNModel(
    model_class=ConvBasicModule,
    model_instantation_args={"size": 64, "channels": 16},
    memory=TensorMemory(capacity=1000, state_shape=(16, 64, 64), device=DQNModel.get_available_device()),
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
        action_id = model.select_action(preprocess_frame(previous_frame, device=model.device), action_space_size=4)
        
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
            done
        )
        model.memory.push(transition)

        model.train_iterations(n_iterations=1, batch_size=32)

        previous_frame = frame
        step_count += 1

    print(f"Episode {episode + 1} finished in {step_count} steps with total reward {total_reward}")

scorecard = env.get_scorecard()
if scorecard:
    print(f"Final Score: {scorecard.score}")