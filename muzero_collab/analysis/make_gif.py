import os

from matplotlib import animation
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from muzero_collab.games.battle import parallel_env, MuZeroConfig, MAPS
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

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def save_frames_as_gif(frames, filename, fps=50):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=fps)
    anim.save(filename, writer='imagemagick', fps=fps)
    plt.close()


def select_action(config, model, stacked_observations, env):
    root, mcts_info = MCTS(config).run(
        model, stacked_observations, env.legal_actions(), env.to_play(), True)

    action = SelfPlay.select_action(root, 0)
    return action


def make_gif(checkpoint_file, output_folder, map='empty', filename=None, fps=50):

    config = MuZeroConfig()

    checkpoint = torch.load(checkpoint_file)

    model = MuZeroNetwork(config)
    model.set_weights(checkpoint['weights'])
    model.to(torch.device('cuda'))

    env = parallel_env(map_name=map)

    observations = env.reset()
    frames = []
    game_histories = {agent: GameHistory(team=RED_TEAM if 'red' in agent else BLUE_TEAM) for agent in env.agents}

    for agent in env.agents:
        game_histories[agent].observation_history.append(observations[agent])

    for t in tqdm(range(250)):
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

    if not filename:
        filename = checkpoint_file.split('/')[-1].split('.')[0]

    gif_filename = os.path.join(output_folder, f'{filename}.gif')

    save_frames_as_gif(frames, gif_filename, fps=fps)


if __name__ == '__main__':
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', help='Filepath to checkpoint file or folder of checkpoints')
    parser.add_argument('--output', default=None, help='Folder to save generated gif file(s) to (Default: <checkpoint folder>/gifs)')
    parser.add_argument('--filename', default=None, help='Filename for output gif (Default: checkpoint filename)')
    parser.add_argument('--fps', type=int, default=50, help='Frames per second of gif (Default: 50)')
    parser.add_argument('--map', default='empty', choices=MAPS.keys(), help='Map to generate gif in (Default: empty)')

    args = parser.parse_args()

    if not args.output:
        if os.path.isdir(args.checkpoint):
            args.output = os.path.join(args.checkpoint, 'gifs')
        else:
            output_path = os.path.dirname(args.checkpoint)

            if not output_path:
                raise Exception(f'Model Checkpoint path should contain at least relative directory reference: {args.checkpoint}')

            args.output = output_path

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.checkpoint):
        checkpoint_files = glob(os.path.join(args.checkpoint, '*.checkpoint'))

        print(f'Found {len(checkpoint_files)} checkpoint files in {args.checkpoint}.')
        print('Generating gifs now...')

        for checkpoint_file in tqdm(checkpoint_files):
            make_gif(checkpoint_file, args.output, filename=args.filename, fps=args.fps, map=args.map)

    else:
        print(f'Generating file for {args.checkpoint}...')

        make_gif(args.checkpoint, args.output, filename=args.filename, fps=args.fps, map=args.map)
