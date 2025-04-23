import numpy as np
from enum import IntEnum
import logging
from models import ActionTypes, EmotionTypes

logging.getLogger().setLevel(logging.INFO)

class Agent:
    def __init__(self):
        self.agent_emotion = None
        self.agent_position = None

    @property
    def position(self):
        return self.agent_position

    @position.setter
    def position(self, value):
        self.agent_position = value

    @property
    def emotion(self):
        if self.agent_emotion is None:
            self.agent_emotion = EmotionTypes.IDLE
        return self.agent_emotion

    @emotion.setter
    def emotion(self, value):
        self.agent_emotion = value

    def take_action(self, state, action_space):
        # set agent position
        self.position = state
        logging.info(f"Agent emotion: {self.emotion.name}")
        return np.random.choice(action_space)