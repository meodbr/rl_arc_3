import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ft09", render_mode=None)

# See available actions
print(env.action_space)
print(type(env.action_space))
for act in env.action_space:
    print(type(act))
    print(act.action_data)
    print(act.action_type)
    print(act.value)
    print(act.is_complex())

# Take an action
obs = env.step(GameAction.ACTION1)

# Check your scorecard
print(arc.get_scorecard())

action = GameAction.from_id(6)
print(action)
print(action.action_data)
action.set_data({"x": 3, "y": 42})
print(action.action_data)