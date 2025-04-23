import numpy as np
from enum import IntEnum
import logging
from models import ActionTypes, EmotionTypes

logging.getLogger().setLevel(logging.INFO)

class Agent:
    def __init__(self):
        self.agent_emotion = EmotionTypes.IDLE

    def get_emotion(self):
        return self.agent_emotion

    def take_action(self, state, action_space):
        logging.info(f"Agent emotion: {self.agent_emotion.name}")
        return np.random.choice(action_space)