import random

import numpy as np
from pettingzoo.magent import battlefield_v2
from pettingzoo.utils import random_demo

env = battlefield_v2.parallel_env()
obs = env.reset()

NUM_STEPS = 500

def sample_action(env, agent, obs):

    if not env.agents:
        action = None
    elif isinstance(obs, dict) and 'action_mask' in obs:
        action = random.choice(np.flatnonzero(obs['action_mask']))
    else:
        action = env.action_spaces[agent].sample()

    return action

for s in range(NUM_STEPS):
    print(f'--- TURN {s} ---')

    env.render()

    actions = {agent: sample_action(env, agent, obs[agent]) for agent in env.agents}
    obs, rewards, done, infos = env.step(actions)

env.close()
