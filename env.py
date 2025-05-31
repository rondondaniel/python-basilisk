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
        self.observation = np.zeros((self.box_size, self.box_size), dtype=np.int32)
        self.done = False
        self.agent_position = None

    def reset(self):
        self.observation = np.zeros((self.box_size, self.box_size), dtype=np.int32)
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

    def get_observation(self):
        """Get observation of the environment"""
        """Tool that allows the agent to observe its environment
        
        Args:
            input_str: Not used, but required by LangChain tool interface
            env: Optional environment to observe
            
        Returns:
            String describing what the agent observes in its environment
        """
        try:
            # Create a description of what the agent sees
            food_positions = np.argwhere(self.observation == CellTypes.FOOD)
            water_positions = np.argwhere(self.observation == CellTypes.WATER)
            
            description = f"You are at position {self.position}. "
            
            # Describe food and water relative to agent position
            if len(food_positions) > 0:
                food_pos = tuple(food_positions[0])
                food_direction = self._get_relative_direction(self.position, food_pos)
                description += f"There is food {food_direction} at position {food_pos}. "
            
            if len(water_positions) > 0:
                water_pos = tuple(water_positions[0])
                water_direction = self._get_relative_direction(self.position, water_pos)
                description += f"There is water {water_direction} at position {water_pos}. "
            
            # Describe boundaries
            grid_size = self.state.shape[0]
            if self.position[0] == 0:
                description += "You are at the north edge. "
            if self.position[0] == grid_size - 1:
                description += "You are at the south edge. "
            if self.position[1] == 0:
                description += "You are at the west edge. "
            if self.position[1] == grid_size - 1:
                description += "You are at the east edge. "
                
            return description
            
        except Exception as e:
            logging.error(f"Error in observe_tool: {str(e)}")
            return f"Error observing environment: {str(e)}"
    
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