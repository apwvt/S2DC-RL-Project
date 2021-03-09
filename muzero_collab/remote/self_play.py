import time

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
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_histories = self.play_game(
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
                for game_history in game_histories:
                    replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_histories = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                episode_length = len(list(game_histories.values())[0]) - 1
                total_reward = sum(sum(gh.reward_history) for gh in game_histories)
                mean_value = numpy.mean(numpy.mean([value for value in gh.root_values if value]) for gh in game_histories)

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": episode_length,
                        "total_reward": total_reward,
                        "mean_value": mean_value
                    }
                )

                if 1 < len(self.config.players):
                    red_reward = sum(sum(gh.reward_history) for gh in game_histories if gh.team == RED_TEAM)
                    blue_reward = sum(sum(gh.reward_history) for gh in game_histories if gh.team == BLUE_TEAM)

                    shared_storage.set_info.remote({'red_reward': red_reward, 'blue_reward': blue_reward})

                    # NOTE: below may be broken given our to-play
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
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """

        assert self.game.agents is not None, 'MuZero implementation not refactored for single player games'
        game_histories = {agent: GameHistory(team=RED_TEAM if 'red' in agent else BLUE_TEAM) for agent in self.game.agents}

        observations = self.game.reset()

        # appending initial values to agents' game histories
        for agent in self.game.agents:
            game_histories[agent].action_history.append(0)
            game_histories[agent].observation_history.append(observations[agent])
            game_histories[agent].reward_history.append(0)
            game_histories[agent].to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (not done and len(list(game_histories.values())[0].action_history) <= self.config.max_moves):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."

                # determine next action for every agent
                actions = {}
                roots = {}
                for agent in self.game.agents:
                    game_history = game_histories[agent]

                    stacked_observations = game_history.get_stacked_observations(-1, self.config.stacked_observations)

                    # Choose the action
                    if opponent == "self" or muzero_player == self.game.to_play():
                        root, mcts_info = MCTS(self.config).run(
                            self.model,
                            stacked_observations,
                            self.game.legal_actions(),
                            self.game.to_play(),
                            True,
                        )
                        action = self.select_action(
                            root,
                            temperature
                            if not temperature_threshold
                            or len(game_history.action_history) < temperature_threshold
                            else 0,
                        )

                        if render:
                            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                            print(
                                f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                            )
                    else:
                        action, root = self.select_opponent_action(
                            opponent, stacked_observations
                        )

                    actions[agent] = action
                    roots[agent] = root

                observations, rewards, dones, infos = self.game.step(action)

                # storing updates in each agent's game history
                for agent in self.game.agents:
                    game_histories[agent].store_search_statistics(roots[agent], self.config.action_space)
                    game_histories[agent].action_history.append(actions[agent])
                    game_histories[agent].observation_history.append(observations[agent])
                    game_histories[agent].reward_history.append(rewards[agent])
                    game_histories[agent].to_play_history.append(self.game.to_play())

                # environment is done if all agents are done
                done = all(d for d in dones.values())

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

        return game_histories

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

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
