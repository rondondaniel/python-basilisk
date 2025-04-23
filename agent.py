import numpy as np

class Agent:
    def __init__(self):
        pass  # No placement logic here

    def take_action(self, action_space):
        return np.random.choice(action_space)