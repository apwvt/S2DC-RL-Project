import copy

import numpy
import ray


@ray.remote
class ReplayBuffer:
    """
    Runs in a dedicated thread containing the game histories that are used for training.
    Other threads can access the buffer to store new games generated through self-play and
    also sample games from the buffer for training the model.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint['num_played_games']
        self.num_played_steps = initial_checkpoint['num_played_steps']
        self.total_samples = sum(len(game_history.root_values) for game_history in self.buffer.values())

        if self.total_samples != 0:
            print(f'Replay Buffer initialized with {self.total_samples} samples ({self.num_played_games} games).')

        numpy.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoids read-only access on array when loading from disk
                game_history.priorities = numpy.copy(game_history.priorities)

            else:
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = numpy.abs(root_value - self.compute_target_value(game_history, i) ** self.config.PER_alpha)
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype='float32')
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        # only keep the buffer size (number of games) to what config says
        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        # if using shared storage, update the number of games and steps
        if shared_storage:
            shared_storage.set_info.remote('num_played_games', values=self.num_played_games)
            shared_storage.set_info.remote('num_played_steps', values=self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def get_batch(self, mapname="empty"):
        index_batch = []
        observation_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        policy_batch = []
        gradient_scale_batch = []

        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size, mapname=mapname):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(game_history, game_pos)

            index_batch.append([game_id, game_pos])
            observation_batch.append(game_history.get_stacked_observations(game_pos, self.config.stacked_observations))
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                    [min(self.config.num_unroll_steps, len(game_history.action_history) - game_pos)] * len(actions))

            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype='float32') / max(weight_batch)

        return (
                index_batch,
                (
                    observation_batch,
                    action_batch,
                    value_batch,
                    reward_batch,
                    policy_batch,
                    weight_batch,
                    gradient_scale_batch,
                    ),
                )

    def sample_game(self, force_uniform=False):
        game_prob = None

        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype='float32',
            )

            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))

        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, mapname="empty", force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []

            for game_id, game_history in self.buffer.items():
                '''
                if game_history.map != mapname:
                    continue
                '''

                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)

            game_probs = numpy.array(game_probs, dtype='float32')
            game_probs /= numpy.sum(game_probs)

            game_prob_dict = {game_id: prob for game_id, prob in zip(game_id_list, game_probs)}

            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)
        else:
            avail_games = []
            for game_id, game_history in self.buffer.items():
                '''
                if game_history.map != mapname:
                    continue
                '''

                avail_games.append(game_id)

            selected_games = numpy.random.choice(avail_games, n_games)
            game_prob_dict = {}

        ret = [(game_id, self.buffer[game_id], game_prob_dict.get(game_id)) for game_id in selected_games]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        position_prob = None

        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            if next(iter(self.buffer)) <= game_id:
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(game_pos + len(priority), len(self.buffer[game_id].priorities))

                self.buffer[game_id].priorities[start_index:end_index] = priority[:end_index - start_index]

                self.buffer[game_id].game_priority = numpy.max(self.buffer[game_id].priorities)

    def compute_target_value(self, game_history, index):
        bootstrap_index = index + self.config.td_steps
        value = 0

        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )

            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index] == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount ** self.config.td_steps

        for i, reward in enumerate(game_history.reward_history[index + 1 : bootstrap_index + 1]):
            value += (
                reward
                if game_history.to_play_history[index] == game_history.to_play_history[index + i]
                else -reward
            ) * self.config.discount ** i

        return value

    def make_target(self, game_history, state_index):
        target_values, target_rewards, target_policies, actions = [], [], [], []

        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(
                    [1 / len(game_history.child_visits[0]) for _ in range(len(game_history.child_visits[0]))]
                )
                actions.append(game_history.action_history[current_index])
            else:
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append(
                    [1 / len(game_history.child_visits[0]) for _ in range(len(game_history.child_visits[0]))]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions
