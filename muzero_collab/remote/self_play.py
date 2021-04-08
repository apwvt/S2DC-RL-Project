import time
import random

import numpy
import ray
import torch

from muzero_collab.models import MuZeroNetwork
from muzero_collab.utils import GameHistory
from muzero_collab.utils import MCTS
from muzero_collab.utils.constants import RED_TEAM, BLUE_TEAM


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        """
        Set up this self play thread and its networks.
        """

        self.config = config

        self.game_dict = { mp: Game(map_name=mp) for mp in self.config.training_maps }

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize alpha network
        self.model = MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

        # Initialize beta network
        self.model_beta = MuZeroNetwork(self.config)
        self.model_beta.set_weights(initial_checkpoint["weights_beta"])
        self.model_beta.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_beta.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        """
        Main body of this self play worker thread: continuously plays games until
        the program terminates or the training step limit is reached.

        This function is also responsible for coordinating and recording training
        metrics, such as mean reward and state value.
        """
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            self.model_beta.set_weights(ray.get(shared_storage.get_info.remote("weights_beta")))

            if not test_mode:
                alpha_histories, beta_histories, game_steps = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                # save each individual game_history to the replay_buffer, counts each agent's move as individual steps
                for game_history in alpha_histories:
                    replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                alpha_histories, beta_histories, game_steps = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                alpha_reward = sum(sum(gh.reward_history) for gh in alpha_histories)
                mean_alpha_reward = alpha_reward / len(alpha_histories)
                mean_alpha_value = numpy.mean([numpy.mean([value for value in gh.root_values if value]) for gh in alpha_histories])

                beta_reward = sum(sum(gh.reward_history) for gh in beta_histories)
                mean_beta_reward = beta_reward / len(beta_histories)
                mean_beta_value = numpy.mean([numpy.mean([value for value in gh.root_values if value]) for gh in beta_histories])

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        'episode_length': game_steps,
                        'alpha_reward': alpha_reward,
                        'mean_alpha_reward': mean_alpha_reward,
                        'mean_alpha_value': mean_alpha_value,
                        'beta_reward': beta_reward,
                        'mean_beta_reward': mean_beta_reward,
                        'mean_beta_value': mean_beta_value,
                    }
                )

                if 1 < len(self.config.players):

                    # NOTE: below may be broken given our to-play, skipping for now
                    pass

                    game_history = list(game_histories.values())[0]
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each move.

        Before moving, each agent runs a fixed number of steps of a Monte Carlo Tree Search
        to approxmiate the value and reward of each of the 21 possible actions.  The best
        move found through this process (with a fuzz factor for exploration) becomes the agent's
        actual move.
        """

        game_mapname = random.choice(list(self.game_dict.keys()))
        game = self.game_dict[game_mapname]

        # choose which team the alpha model will control
        # TODO: add number of teams to config
        alpha_team = random.randint(0, 1)

        observations = game.reset()

        game_histories = {agent: GameHistory(team=RED_TEAM if 'red' in agent else BLUE_TEAM, mapname=game_mapname) for agent in game.agents}

        # appending initial values to agents' game histories
        for agent in game.agents:
            game_histories[agent].action_history.append(0)
            game_histories[agent].observation_history.append(observations[agent])
            game_histories[agent].reward_history.append(0)
            game_histories[agent].to_play_history.append(game.to_play())

        done = False
        game_steps = 0

        if render:
            game.render()

        with torch.no_grad():
            while (not done) and (game_steps <= self.config.max_moves):
                assert (
                    len(numpy.array(list(observations.values())[0]).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(list(observations.values())[0]).shape)} dimensionnal. Got observation of shape: {numpy.array(list(observations.values())[0]).shape}"
                assert (
                    numpy.array(list(observations.values())[0]).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(list(observations.values())[0]).shape}."

                # determine next action for every agent
                actions = {}
                roots = {}
                for agent in game.agents:
                    game_history = game_histories[agent]

                    this_model = self.model if game_history.team == alpha_team else self.model_beta

                    stacked_observations = game_history.get_stacked_observations(-1, self.config.stacked_observations)

                    # Choose the action
                    if opponent == "self" or muzero_player == game.to_play():
                        root, mcts_info = MCTS(self.config).run(
                            this_model,
                            stacked_observations,
                            game.legal_actions(),
                            game.to_play(),
                            True,
                        )
                        action = self.select_action(
                            root,
                            temperature
                            if (not temperature_threshold
                            or len(game_history.action_history) < temperature_threshold)
                            and game_history.team == alpha_team
                            else 0,
                        )

                        if render:
                            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                            print(
                                f"Root value for player {game.to_play()}: {root.value():.2f}"
                            )

                    actions[agent] = action
                    roots[agent] = root

                observations, rewards, dones, infos = game.step(actions)

                # storing updates in each agent's game history
                for agent in game.agents:
                    game_histories[agent].store_search_statistics(roots[agent], self.config.action_space)
                    game_histories[agent].action_history.append(actions[agent])
                    game_histories[agent].observation_history.append(observations[agent])
                    game_histories[agent].reward_history.append(rewards[agent])
                    game_histories[agent].to_play_history.append(game.to_play())

                # environment is done if all agents are done
                done = not game.agents
                game_steps += 1

                if render:
                    #print(f"Played action: {game.action_to_string(action)}")
                    game.render()

        alpha_histories = [gh for gh in game_histories.values() if gh.team == alpha_team]
        beta_histories = [gh for gh in game_histories.values() if gh.team != alpha_team]
        return alpha_histories, beta_histories, game_steps

    def close_game(self):
        """
        Clean up the game environments when this program is closing.
        """
        for game in self.game_dict.values():
            game.close()

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action
