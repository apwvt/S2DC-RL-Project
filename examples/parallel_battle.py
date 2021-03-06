import random
import sys
import time

import numpy as np
from pettingzoo.magent import battle_v2
from pettingzoo.utils import random_demo

"""
This is an example of running the Battle environment in parallel.

It provides cli functionality to change the numer of steps in the
environment as well as the map size.

Example for 100 steps in a 20x20 environment:
    python examples/parallel_battle.py 100 20
"""

NUM_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MAP_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 15

height = MAP_SIZE
width = MAP_SIZE

env = battle_v2.parallel_env(map_size=MAP_SIZE)

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
