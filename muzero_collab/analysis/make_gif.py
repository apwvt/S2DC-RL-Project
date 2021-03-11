import sys

from matplotlib import animation
import matplotlib.pyplot as plt
import torch

from muzero_collab.games.battle import parallel_env, MuZeroConfig
from muzero_collab.models import MuZeroNetwork
from muzero_collab.utils import GameHistory, MCTS
from muzero_collab.utils.constants import RED_TEAM, BLUE_TEAM
from muzero_collab.remote import SelfPlay

"""
Ensure you have imagemagick installed with
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""


def save_frames_as_gif(frames, filename, path='../gifs'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def select_action(config, model, stacked_observations, env):
    root, mcts_info = MCTS(config).run(
        model, stacked_observations, env.legal_actions(), env.to_play(), True)

    action = SelfPlay.select_action(root, 0)
    return action


if __name__ == '__main__':
    checkpoint_file = sys.argv[1]
    gif_filename = sys.argv[2]

    config = MuZeroConfig()

    checkpoint = torch.load(checkpoint_file)

    model = MuZeroNetwork(config)
    model.set_weights(checkpoint['weights'])
    model.to(torch.device('cuda'))

    env = parallel_env()

    observations = env.reset()
    frames = []
    game_histories = {agent: GameHistory(team=RED_TEAM if 'red' in agent else BLUE_TEAM) for agent in env.agents}

    for agent in env.agents:
        game_histories[agent].observation_history.append(observations[agent])

    for t in range(250):
        #Render to frames buffer
        frames.append(env.render(mode="rgb_array"))

        actions = {}

        for agent in env.agents:
            game_history = game_histories[agent]
            stacked_observations = game_history.get_stacked_observations(-1, config.stacked_observations)

            action = select_action(config, model, stacked_observations, env)

            actions[agent] = action

        observations, rewards, _, _ = env.step(actions)

        if not env.agents:
            break

    env.close()
    save_frames_as_gif(frames, gif_filename)
