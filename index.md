## Overview

The past few years have seen an explosion of applications of Single-Agent and Multi-Agent Reinforcement Learning over fields ranging from board games to drone control to social interaction.  One of the new architectures deployed to meet this emerging need is MuZero, which broke new ground in single-agent environments such as Chess and Go.  However, previous work has not attempted to adapt MuZero to games with more than one agent on each team.

To evaluate the suitability of MuZero for this family of applications, we spent our semester data-centric computing project adapting the algorithm for multi-agent games.  Beginning with the general MuZero implementation by [Duvaud and Hainaut](https://github.com/werner-duvaud/muzero-general), we modified parts of the algorithm's structure, particularly the Monte-Carlo Tree Search implementation, in a way that allowed for the efficient prediction of multiple agent actions.  We then trained a single model to play a six-on-six adversarial game derived from [PettingZoo's MAgent package](https://www.pettingzoo.ml/magent/battle) over ~35k training steps.

https://user-images.githubusercontent.com/57198618/117064489-43894780-acf4-11eb-9e22-52eb03831aa0.mp4

### Results
While the algorithm did learn to play the game, the level of strategic sophistication displayed was less than anticipated.  In particular, the MuZero model appeared to perform worse than comparable deep reinforcement learning algorithms over comparable training times, with agents only occasionally making progress in their task (dealing enough damage to score a kill) before the scenario hit its step limit.

Overall, we were forced to conclude that multi-agent MuZero is not an improvement on present multi-agent reinforcment learning algorithms.

### Future Work
We leave to future work a number of avenues for exploration.  Given our limited access to computing hardware, we deliberately chose very small network and batch sizes; our model may simply be underpowered for the environment we chose.  In particular, the small size of the encoded state used in the initial and recurrent inferences is a suspected bottleneck in model learning capacity.  We also found that the agents' behavior was greatly affected by the reward structure, and had to experiment with several reward schemes to find one that would not regress to degenerate strategies; there may exist a reward structure that produces better team-based performance with the same architecture, possibly by rewarding the agents as a group.
