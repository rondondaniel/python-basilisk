"""
Agent module that interfaces with AgentBrain
"""
import logging
from agent_brain import AgentBrain

logging.getLogger().setLevel(logging.INFO)

class Agent:
    def __init__(self):
        self.agent_brain = AgentBrain()

    @property
    def position(self):
        return self.agent_brain.position

    @property
    def emotion(self):
        return self.agent_brain.emotion

    @property
    def state(self):
        return self.agent_brain.state

    def initialize_state_from_env(self, env):
        """Initialize the agent's state from an environment"""
        self.agent_brain.init_state_from_env(env)
        return self.agent_brain.state

    def observe(self, env):
        """Use the agent's observe tool to get a description of the environment
        
        Args:
            env: The environment to observe
        """
        return self.agent_brain._observe_tool("", env)

    def process_query(self, query, env=None):
        """Process a query through the agent brain
        
        Args:
            query: The query to process
            env: Optional environment to use for context
        """
        return self.agent_brain.run(query, env)

    def take_action(self, direction, env):
        """Take a movement action
        
        Args:
            direction: The direction to move in
            env: The environment to act upon
        """
        direction_json = f'{{"direction": "{direction}"}}'  
        return self.agent_brain._move_tool(direction_json, env)

    def set_emotion(self, emotion_name):
        """Set the agent's emotion"""
        emotion_json = f'{{"emotion": "{emotion_name}"}}'  
        return self.agent_brain._emotion_tool(emotion_json)
