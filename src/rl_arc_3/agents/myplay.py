import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

# Take a few actions
for _ in range(10):
    obs = env.step(GameAction.ACTION1)
    print(f"Action taken, received observation: {obs}")
    print(f"Dict: {obs.__dict__}")
    print(f"Frame: {obs.frame[-1]}")
    print(f"Frame shape: {obs.frame[-1].shape}")

print(arc.get_scorecard())