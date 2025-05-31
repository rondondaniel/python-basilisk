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

    def _process_query(self, query, observation: np.ndarray):
        """Process a query through the agent brain
        
        Args:
            query: The query to process
            observation: The observation to act upon
        """
        return self.agent_brain.run(query, observation)

    def take_action(self, observation):
        """Take a movement action
        
        Args:
            observation: The observation to act upon
        """
        response = self._process_query("What should I do next?", observation)
        
        return response['action']

    def set_emotion(self, emotion_name):
        """Set the agent's emotion"""
        emotion_json = f'{{"emotion": "{emotion_name}"}}'  
        return self.agent_brain._emotion_tool(emotion_json)
