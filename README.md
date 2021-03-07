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

## MuZero Implementation

The Implementation of MuZero is done in PyTorch and uses Ray to improve training performance through having multiple
self-play actors running in various threads contributing data to a shared replay buffer.

The initial code organization and implementation is based on 
[Werner Duvaud's general MuZero implementation](https://github.com/werner-duvaud/muzero-general)

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

