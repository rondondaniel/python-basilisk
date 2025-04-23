from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from models import ActionTypes, CellTypes

class BoardEnv(Env):
    def __init__(self):
        super().__init__()
        self.box_size = 8
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=3, shape=(self.box_size, self.box_size), dtype=np.int32)
        self.state = np.zeros((self.box_size, self.box_size), dtype=np.int32)
        self.done = False
        self.agent_position = None

    def reset(self):
        self.state = np.zeros((self.box_size, self.box_size), dtype=np.int32)
        self.done = False
        # Randomly place two positions to generate water and food
        flat_indices = np.random.choice(self.box_size * self.box_size, 2, replace=False)
        food_pos = np.unravel_index(flat_indices[0], (self.box_size, self.box_size))
        water_pos = np.unravel_index(flat_indices[1], (self.box_size, self.box_size))
        self.state[food_pos] = CellTypes.FOOD    # food
        self.state[water_pos] = CellTypes.WATER  # water
        # Place agent in an empty cell
        empty_cells = np.argwhere(self.state == CellTypes.EMPTY)
        idx = np.random.choice(len(empty_cells))
        self.agent_position = tuple(empty_cells[idx])
        self.state[self.agent_position] = CellTypes.AGENT
        return self.state

    def _perform_action(self, action):
        match action:
            case ActionTypes.LEFT:  # left
                new_position = (self.agent_position[0], self.agent_position[1] - 1)
            case ActionTypes.RIGHT:  # right
                new_position = (self.agent_position[0], self.agent_position[1] + 1)
            case ActionTypes.UP:  # up
                new_position = (self.agent_position[0] - 1, self.agent_position[1])
            case ActionTypes.DOWN:  # down
                new_position = (self.agent_position[0] + 1, self.agent_position[1])
            case ActionTypes.STAY:  # stay
                new_position = self.agent_position
            case _:
                raise ValueError("Invalid action")
        return new_position

    def step(self, action):
        # move agent from its actual position to new position based on one of 5
        # actions: left, right, up, down, stay
        new_position = self._perform_action(action)
        # Erase old position
        self.state[self.agent_position] = CellTypes.EMPTY
        # Move agent to new position
        self.agent_position = new_position
        self.state[self.agent_position] = CellTypes.AGENT
        self.done = True

        assert self.observation_space.contains(self.state), "State contains invalid values!"
        return self.state, 0, self.done, {}

    def render(self, mode='human'):
        print(self.state)

        assert self.observation_space.contains(self.state), "State contains invalid values!"
        return self.state, 0, self.done, {}

    def render(self, mode='human'):
        print(self.state)