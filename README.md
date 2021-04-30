# MuZero Collaborative Agents

This repo contains our senior capstone project for Virginia Tech 2021. We are looking into training
collaborative agents through self-play using the MuZero algorithm. 

## Environment

For our environment, we are using a slightly modified version of PettingZoo's Battle environment. Our
modifications include changes into how the agents are spawned such that teams are more likely to be balanced.
The adjustments made here are not final nor are they tested, but thus far have produced balanced teams on a 15x15
map. 

The environment was also adjusted to allow for a map layout to be provided. This allows for walls to be added as in
other PettingZoo environments. 

For a simple example of our modified environment in action, look at `examples/custom_battle.py`.

## Project Structure

The project's code lives within the `muzero_collab` folder. This is done because the project is set up as a pip
installable package to allow for friendlier importing. See the [Development](#Development) section for the commands
to locally install and run the project.

The `examples` folder contains brief code snippets demonstrating simple patterns and were used for experimentation
in the project's earlier stages such as familiarzing ourselves with the returns from the environment.

Each folder and file contained within the package is explained in the following sections. For a more in-depth discussion
on the flow of information throughout the files, see the description found in the 
[original implementation](https://github.com/werner-duvaud/muzero-general).

### `analysis`

Overall, the analysis folder contains scripts that are ran to provide some sort of analysis
feature throughout training or afterwards. 

#### `copier.py`

This script facilitates checkpointing the model throughout training by creating periodic
copies of the current training checkpoint file. This strategy was chosen over building 
checkpointing into the training script itself because it was determined to be simpler than
adding another component that must be managed during training.

The script is ran as a cli interface. Arguments include: the experiment folder to checkpoint,
the frequency at which to checkpoint, and whether to immediately generate a gif or not. The
full help menu can be viewed by running the script with the `-h` flag.

#### `make_gif.py`

The `make_gif.py` script provides a cli interface to generate a gif of the given model. It allows
for the map to be selected as well as the playback speed to be changed.

For a full description of how to use the script, run it with the `-h` flag.

### `games`

The `games` folder contains the environments that the model can act within. Upon instaniation,
various classes such as Self Play and the Trainer create their own versions of the environment
for models to play in. 

For each environment, the default configuration is also described directly in the file. This
configuration contains many different features a few of which include: network type, network
architectures, and many other training parameters.

#### `abstract_game.py`

`abstract_game.py` defines what every game or environment must contain. It acts as an abstract class
that all other environments inherit from. With this the necessary behaviors an environment must have
are enforced.

#### `battle.py`

`battle.py` contains a slightly modified version of [PettingZoo's Battle](https://www.pettingzoo.ml/magent/battle)
environment. Prior to beginning this project. The PettingZoo version did not properly balance teams with the map size
sometimes leading to uneven teams. This was rudimentarily enforced in our version for our map size (12x12). Additionally,
the ability to define obstacles within the map was also included.

The file also contains the default configuration for the environment. Because a model has not been successfully trained
in the environment yet, these defaults are very much experimental and should not be trusted.

### `models`

The `models` folder contains the different model architectures' implementations as well
as some helper functions. 

#### `abstract.py`

`abstract.py` contains an abstract implementation of a model defining the necessary functions 
for all subclasses. These functions enforce the ability to create the initial inference, recurrent
infereces, as well as importing and exporting the model's weights.

#### `fully_connected.py`

`fully_connected.py` provides a fully connected architecture for each of the models contained
within MuZero. This file defines the entire MuZero architecture using fully connected networks.
The fully connected networks are created using multi-layered perceptrons defined in the `mlp.py`
file also in this folder.

#### `mlp.py`

`mlp.py` contains the standard implementation of a multi-layered perceptron which is used
within the fully connected architecture.

#### `muzero_network.py`

`muzero_network.py` is used to create a model given a configuration. From the configuration,
the network architecture is recognized as fully connected or residual, and the correct
MuZero construction is returned based off the remaining configuration.

#### `residual.py`

`residual.py` defines the MuZero architecture using residual networks within. So far, the fully
connected architecture has been used for all training and this architecture has not bee explored.

#### `utils.py`

`utils.py` contains various utility functions used throughout models and when interacting with
models such as ensuring a dictionary is on the cpu as well as functions to scale values accordingly.

### `remote`

The `remote` folder contains all the classes that are ran "remotely" on different threads through the `ray`
package. This allows for parallelization and increased efficiency in training. 

#### `reanalyse.py`

`reanalyse.py` contains the Reanalyse class as described by the authors. Reanalyse constantly
reviews games stored within the Replay Buffer to update the values with the latest model's 
predictions. This is done to prevent incorrect values assigned by past models to negatively
affect training. Overall, this phases increases the Replay Buffer's efficiency in regards to
training by ensuring it is more up-to-date.

#### `replay_buffer.py`

`replay_buffer.py` defines the memory used to store play-throughs of the environment and the
results which are then used for training. Once Self Play has completed a game, the history will
be stored within the Replay Buffer. The Trainer queries the Replay Buffer for batches of games
to train on. Reanalyse is also constantly using the latest model to update values of game histories
stored within the Replay Buffer.

#### `self_play.py`

`self_play.py` houses the code that generates the games stored within the Replay Buffer. The latest
models will be loaded from the Shared Memory and used to play a game. Once the game has completed,
the game histories will be sent to the Replay Buffer to later be used by the trainer. Typically,
multiple threads are running Self Play to efficiently create games to fill the Replay Buffer with.

#### `shared_storage.py`

`shared_storage.py` defines the Shared Storage which is essentially a wrapper around a dictionary to
allow for simple and consistent access of information across threads. The Shared Storage contains
the current state of training including the current training steps, the model weights, and optimizer states.

When checkpointing the model, the information stored within the Shared Storage is saved. This has all
the information needed to continue training where it left off.

### `utils`

`utils.py` contains various utility classes or smaller classes that appear in many places throughout
the code base. 

#### `constants.py`

`constants.py` contains constants used throughout the codebase. Currently, this is just the integer
values used to represent the red and blues teams.

#### `cpu_actor.py`

`cpu_actor.py` is a workaround in ray to force DataParallel objects to remain on the CPU
even if there is a GPU available that they would default to. This is used when training is
began for the first time.

#### `game_history.py`

`game_history.py` implements the Game History class which stores what occurred during a game.
A Game History records a game from a single agent's view. Therefore, for one environment cycle,
six Game Histories will be generated - one for each agent on the alpha team. The Game History
contains everything about the game including the team, every observation, subsequent action, and 
the associated rewards. These are used as training data for the model to improve.

#### `mcts.py`

`mcts.py` contains the Monte Carlo Tree Search algorithm used within MuZero to simulate the 
environment further and develop a tree where eventually the next action is selected from. 

#### `minmax.py`

`minmax.py` defines a class used within the Monte Carlo Tree Search that holds the minimum
and maximum values of the tree. 

### `muzero.py`

`muzero.py` is the overall training class. This class starts the `ray` process and all necessary
actors. Once the actors have begun training, a logging loop is entered to record training progress
to TensorBoard. Currently this loop logs on a timestep, there are plans to convert the results
to be logged on each training epoch instead. This will allow for more standard "Y vs training epoch"
graphs to be analyzed which better represent training progress.

        >>> muzero = MuZero("ignored")
        >>> muzero.train()

To simplify changing the training configuration, the `MuZero` constructor also takes a `config` keyword
argument that can either be a dictionary or a custom configuration object as defined in the game class.
If a dictionary is given, the keys are strings matching the configuration parameters that they
will overwrite. Any configuration parameters not overwritten will remain at the default value.

## MuZero Implementation

The Implementation of MuZero is done in PyTorch and uses Ray to improve training performance through having multiple
self-play actors running in various threads contributing data to a shared replay buffer.

The initial code organization and implementation is based on 
[Werner Duvaud's general MuZero implementation](https://github.com/werner-duvaud/muzero-general)

## How to Train

The model can be trained by simply running the following two lines from a python interpreter
after setting up the environment (See [Development](#Development)).

        >>> muzero = MuZero("ignored")
        >>> muzero.train()

This will begin training the model and by default the results are stored in a folder roughly 
following the pattern `results/battle/<experiment name>` where the experiment name defaults to
the current date.

### Checkpointing 

Checkpointing is achieved by simply making periodic copies of the `model.checkpoint` file for 
the experiment. The `analysis/copier.py` script facilities this into a cli. It also provides
the ability to automatically generate gifs.

The script can be used as follows which will create a copy by default every 30 minutes.
More information can be found by running the script with the -h flag.

    python analysis/copier.py <path to checkpoint file> 

### Loading Models from Checkpoints

Many times, a model may need to be ran again such as for generating gifs. This requires the model
to be loaded from a checkpoint file. The following code snippet demonstrates how to create
a model from a checkpoint file.

    
    config = MuZeroConfig()

    checkpoint = torch.load(checkpoint_file)

    model = MuZeroNetwork(config)
    model.set_weights(checkpoint['weights'])

First, a default configuration is created

__This is wrong, the config should not be default. The config should be loaded from
the checkpoint file then the model should just be able to be created from it? This should brea
when architecture differs from the default__

## Development

In order to develop and run the code, it is recommended that the package we installed locally. The simplest way of
doing this is to run the following code from the root folder after pulling the repo:

    pip install -e .

This ensures that the proper connections are established for imports and such. The package will be installed in an
editable fashion such that changes to the code will be recognized and another installation will not be needed.

__If receive an error that the `muzero_collab` module cannot be found, you most likely have not installed the 
package locally.__

__When running the code in temporary environments (Docker Containers, Google Collab Notebooks, etc.), this line
will need to be run with each new instance.__

### Docker Container
The code can be run a Docker container using the latest PyTorch image and Docker installed with GPU pass-through. 
Running the following command from the project's root directory will mount the source code into `/code` within the 
container. Upon entering the container, be sure to run the `pip install -e .` command to set up the local installation.
To prevent the container from being deleted to save space when possible, remove the `--rm` flag.

    docker run --gpus all --rm -it -v $(pwd):/code nvcr.io/nvidia/pytorch:21.03-py3

