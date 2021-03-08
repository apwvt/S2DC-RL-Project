import time

import numpy
import ray
import torch

import muzero_collab.models

@ray.remote
class Reanalyze:
    def __init__(self, initial_checkpoint, config):
        self.config = config

        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint['weights'])
        self.model.to(torch.device('cuda') if torch.cuda.is_available() and self.config.reanalyze_on_gpu else 'cpu')
        self.model.eval()

        self.num_reanalyzed_games = initial_checkpoint['num_reanalyzed_games']

    def reanalyze(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote('num_played_names')) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote('training_step')
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote('terminate')
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote('weights')))

            game_id, game_history, _ = ray.get(replay_buffer.sample_game.remote(force_uniform=True))

            if self.config.use_last_model_value:
                observations = [
                    game_history.get_stacked_observations(i, self.config.stacked_observations)
                    for i in range(len(game_history.root_vales))
                ]

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )

                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size
                )

                game_history.reanalyzed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalyzed_games += 1
            shared_storage.set_info.remote('num_reanalyzed_games', self.num_reanalyzed_games)

