import math

from pettingzoo.magent.battle_v2 import _parallel_env

DEFAULT_MAP_SIZE = 15
DEFAULT_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
DEFAULT_MAX_CYCLES = 1000
DEFAULT_KILL_REWARD = 5
DEFAULT_MINIMAP_MODE = True
DEFAULT_REWARD_ARGS = dict(step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2)



def parallel_env(map_size=DEFAULT_MAP_SIZE, max_cycles=DEFAULT_MAX_CYCLES, minimap_mode=DEFAULT_MINIMAP_MODE, map_layout=DEFAULT_MAP, **reward_args):
    env_reward_args = dict(**DEFAULT_REWARD_ARGS)
    env_reward_args.update(reward_args)
    return BattleEnv(map_size, minimap_mode, env_reward_args, max_cycles, map_layout)

class BattleEnv(_parallel_env):
    """Custom Battle Environment that allows for modification of the map through a configuration

    The code to generate the map which adds the players has been modified to include adding barriers as seen in the
    Battlefield environment. An additional change has been made to help balance teams as this is not ensured in the
    base Battle Environment. It is still not certain that teams will always be balanced, but using a map_size of 15
    will provide a consistent 6v6 environment as long as no players collide with barriers.

    It is recommended to use the default map_size of 15 and any custom maps should avoid placing barriers near either
    the left or right edge where teams are spawned.
    """

    def __init__(self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        map_layout
    ):
        self.map_size = map_size
        self._configure_map(map_layout)
        super().__init__(map_size, minimap_mode, reward_args, max_cycles)

    def _configure_map(self, map_layout):
        """Configures the map given a layout

        The map_layout is a 2D array where zeroes represent open spaces and all other values signify a barrier.
        See DEFAULT_MAP for an example.

        Args:
            map_layout: 2D array describing how to setup the map
        """
        assert len(map_layout) == self.map_size
        self.obs_pos = []
        for i, row in enumerate(map_layout):
            assert len(row) == self.map_size, 'all rows in map_layout must have length map_size'
            for j, col in enumerate(row):
                if col:
                    self.obs_pos.append((j, i))

    def legal_actions(self):
        return list(range(21))

    def to_play(self):
        return 0

    def generate_map(self):
        """Generates the map for a run through the environment

        This function is called within the super class's `reset` function to create a new game.
        It is responsible for adding the barriers to the game board as well as spawning each of the
        teams. The size of the teams changes with the size of the map.

        Besides adding the barriers, another slight modification has been made to try and encourage
        balanced teams to be created more often. These changes still do not garuantee for balanced teams.
        """
        # add the barriers
        self.env.add_walls(pos=self.obs_pos, method='custom')

        env, map_size, handles = self.env, self.map_size, self.handles
        width = height = map_size
        init_num = map_size * map_size * 0.04
        gap = 3

        self.leftID, self.rightID = self.rightID, self.leftID

        # Spawn the left (red) team
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        for x in range(0, side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                pos.append([x, y, 0])
        env.add_agents(handles[self.leftID], method="custom", pos=pos)

        # Spawn the right (blue) team
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        hi = width - 1
        lo = width + 1 - side
        for x in range(lo, hi, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                pos.append([x, y, 0])
        env.add_agents(handles[self.rightID], method="custom", pos=pos)
