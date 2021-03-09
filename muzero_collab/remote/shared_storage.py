import copy
import os

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Runs in a dedicated thread storing information to be accessed by multiple other threads such
    as:
        * network weights
        * training status (losses, learning rate, training step)
        * number of games played

    Information is accessed by other threads to complete actions including updating their model
    to the latest weights, fetching data to train the model, and signal when training has completed.

    Fields:
        config: a class containing fields to describe the training configuration.
        checkpoint: dictionary like object to establish where to begin training from and to
                    track training progress - where information is stored
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None):
        """Saves the current checkpoint at the given path or the one found under config.results_path"""
        if not path:
            path = os.path.join(self.config.results_path, "model.checkpoint")

        torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        """Returns a copy of the current checkpoint"""
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        """Gets information about the current checkpoint

        If keys is a string, returns the corresponding value.
        If keys is a list, returns a dictionary with requested keys and their values
        """
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        """Updates the information stored within the checkpoint.

        If keys is a string, will update the key in the checkpoint with the provided value.
        If keys is a dictionary, will update the checkpoint with the dictionary's keys and values.
        """
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
