import random
import sys
import time

import numpy as np
from pettingzoo.magent import battle_v2
from pettingzoo.utils import random_demo

import muzero_collab
from muzero_collab.environment.battle import parallel_env

"""
An example of running the custom Battle environment and modifying 
the map layout from the default. 

Provides cli functionality to change the number of steps in the game.

Example:
    python examples/custom_battle.py 200
"""

custom_env = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

NUM_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50

env = parallel_env(map_layout=custom_env)

obs = env.reset()

def sample_action(env, agent, obs):

    if not env.agents:
        action = None
    elif isinstance(obs, dict) and 'action_mask' in obs:
        action = random.choice(np.flatnonzero(obs['action_mask']))
    else:
        action = env.action_spaces[agent].sample()

    return action

for s in range(1, NUM_STEPS + 1):
    print(f'--- TURN {s} ---')

    env.render()

    actions = {agent: sample_action(env, agent, obs[agent]) for agent in env.agents}
    obs, rewards, done, infos = env.step(actions)
    time.sleep(0.01)

env.close()
