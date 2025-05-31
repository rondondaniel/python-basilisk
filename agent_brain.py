"""
Agent Brain - Core logic for decision-making and environment interaction
Using LangGraph to implement a ReAct agent pattern
"""
import numpy as np
import logging
import json
from models import ActionTypes, EmotionTypes, CellTypes
from typing import Dict, Any, List, Tuple, Annotated, TypedDict, Literal, Union, Optional
import operator
from enum import Enum

# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool, BaseTool
from langchain_ollama import OllamaLLM

# Define state schema for the LangGraph
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    observation: np.ndarray
    position: Optional[Tuple[int, int]]
    emotion: Optional[str]
    action: Optional[int]
    error: Optional[str]

class AgentBrain:
    def __init__(self):
        self._emotion = None
        self._emotions_memory: List[str] = []
        self._position = None  # Store agent position
        self._current_observation = None
        
        # Initialize the LLM
        self._llm = OllamaLLM(
            model="llama3.2:1b",  # 1.2B parameter version
            base_url="http://127.0.0.1:11434",  # Ollama server URL
            temperature=0.7
        )

        self.tools = self._setup_tools()

        self._agent_executor = create_react_agent(
            model=self._llm,
            tools=self.tools,
            response_format=AgentState,
        )

    def _setup_tools(self):
        """Set up the tools for the agent"""
        
        @tool
        def get_emotion() -> str:
            """Get the current emotion of the agent"""
            return f"Current emotion: {self._emotion.name if self._emotion else 'None'}"
        
        @tool
        def set_emotion(emotion_name: str) -> str:
            """Set the agent's emotion (idle, hungry, thirsty)"""
            try:
                emotion_name = emotion_name.upper()
                if hasattr(EmotionTypes, emotion_name):
                    self._emotion = getattr(EmotionTypes, emotion_name)
                    self._emotions_memory.append(self._emotion.name)
                    return f"Emotion set to: {self._emotion.name}"
                else:
                    return f"Unknown emotion: {emotion_name}. Available emotions: idle, hungry, thirsty"
            except Exception as e:
                logging.error(f"Error in set_emotion: {str(e)}")
                return f"Error setting emotion: {str(e)}"
        
        @tool
        def get_direction(direction: Literal['left', 'right', 'up', 'down', 'stay']) -> str:
            """Get the action value for a given direction"""
            direction_map = {
                "left": ActionTypes.LEFT.value,
                "right": ActionTypes.RIGHT.value,
                "up": ActionTypes.UP.value,
                "down": ActionTypes.DOWN.value,
                "stay": ActionTypes.STAY.value
            }
            
            if direction not in direction_map:
                return f"Invalid direction: {direction}. Valid directions: left, right, up, down, stay"
                
            return str(direction_map[direction])
        
        @tool
        def describe_observation() -> str:
            """Describe the current observation"""
            if self._current_observation is None:
                return "No observation available"
            
            return self._get_state_description()
                
        # Store tools for LangGraph
        tools = [get_emotion, set_emotion, get_direction, describe_observation]

        return tools

    def _emotion_tool(self, input_str: str = "") -> str:
        """Tool that returns the agent's current emotion or sets a new emotion
        
        Args:
            input_str: Optional JSON string with format: {"emotion": "emotion_name"}
                       where emotion_name is one of: idle, hungry, thirsty
                       If not provided, returns current emotion
        
        Returns:
            String describing the current emotion
        """
        try:
            # If no input is provided, just return current emotion
            if not input_str:
                if self.emotion is None:
                    return "Current emotion: Not set (None)"
                return f"Current emotion: {self.emotion.name}"
            
            # Parse input to set a new emotion if provided
            input_data = json.loads(input_str) if isinstance(input_str, str) else input_str
            new_emotion_str = input_data.get("emotion", "").upper()
            
            # Validate and set new emotion
            try:
                new_emotion = EmotionTypes[new_emotion_str]
                self.emotion = new_emotion
                logging.info(f"Emotion changed to {new_emotion.name}")
                return f"Emotion set to: {new_emotion.name}"
            except KeyError:
                valid_emotions = [e.name for e in EmotionTypes]
                return f"Invalid emotion: {new_emotion_str}. Valid emotions are: {', '.join(valid_emotions)}"
        
        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide a valid JSON object with 'emotion' key."
        except Exception as e:
            logging.error(f"Error in emotion_tool: {str(e)}")
            return f"Error handling emotion: {str(e)}"

    def _move_tool(self, input_str: str) -> str:
        """Tool that determines the agent's move decision based on ActionTypes
        
        Args:
            input_str: JSON string with format: {"direction": "direction_name"}
                       where direction_name is one of: left, right, up, down, stay
            env: Optional environment to use for the action
            
        Returns:
            String describing the outcome of the move action
        """
        try:
            print("Move tool called with input: %s" % input_str)
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide a valid JSON object with 'direction' key."
        except Exception as e:
            logging.error(f"Error in move_tool: {str(e)}")
            return f"Error processing move: {str(e)}"
            
    def _describe_observation_tool(self) -> str:
        """Tool that describes observation to help the agent to observe its environment"""
        try:
            return self._get_state_description()
        except Exception as e:
            logging.error(f"Error in describe_observation_tool: {str(e)}")
            return f"Error processing describe observation: {str(e)}"
    
    def _get_relative_direction(self, from_pos, to_pos):
        """Get relative direction from one position to another"""
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        
        directions = []
        if row_diff < 0:
            directions.append("north")
        elif row_diff > 0:
            directions.append("south")
            
        if col_diff < 0:
            directions.append("west")
        elif col_diff > 0:
            directions.append("east")
            
        if not directions:
            return "at your current position"
        
        return " and ".join(directions)
    
    def _get_state_description(self) -> str:
        """Get a description of the current state without needing an environment"""
        if self._current_observation is None or self._position is None:
            return "No state information available."
        
        # Create a description based on the current observation
        food_positions = np.argwhere(self._current_observation == CellTypes.FOOD)
        water_positions = np.argwhere(self._current_observation == CellTypes.WATER)
        
        description = f"You are at position {self._position}. "
        
        # Describe food and water relative to agent position
        if len(food_positions) > 0:
            food_pos = tuple(food_positions[0])
            food_direction = self._get_relative_direction(self._position, food_pos)
            description += f"There is food {food_direction} at position {food_pos}. "
        
        if len(water_positions) > 0:
            water_pos = tuple(water_positions[0])
            water_direction = self._get_relative_direction(self._position, water_pos)
            description += f"There is water {water_direction} at position {water_pos}. "
        
        # Describe boundaries
        grid_size = self._current_observation.shape[0]
        if self._position[0] == 0:
            description += "You are at the north edge. "
        if self._position[0] == grid_size - 1:
            description += "You are at the south edge. "
        if self._position[1] == 0:
            description += "You are at the west edge. "
        if self._position[1] == grid_size - 1:
            description += "You are at the east edge. "
            
        return description

    def _update_emotion_based_on_previous_emotions(self, emotions_memory: List[str]) -> None:
        """Update the agent's emotion based on previous emotions
        
        Args:
            emotions_memory: List of previous emotions
        """
        if not emotions_memory or len(emotions_memory) < 3:
            return
            
        # Get last 3 emotions
        last_emotions = emotions_memory[-3:]
        
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion in last_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # If we've been the same emotion for 3 turns, consider changing
        if len(emotion_counts) == 1 and list(emotion_counts.values())[0] == 3:
            # If we're IDLE, randomly choose between HUNGRY and THIRSTY
            if last_emotions[0] == EmotionTypes.IDLE.name:
                import random
                self._emotion = random.choice([EmotionTypes.HUNGRY, EmotionTypes.THIRSTY])
            # If we're HUNGRY or THIRSTY for too long, go back to IDLE
            else:
                self._emotion = EmotionTypes.IDLE
        
    
    def _extract_action_from_message(self, message_content: str) -> Optional[int]:
        """Extract an action from the agent's message
        
        Args:
            message_content: The content of the agent's message
            
        Returns:
            The action value (int) from ActionTypes, or None if no action found
        """
        # Look for directional keywords in the message
        message = message_content.lower()
        
        # Check for direction mentions
        if "move left" in message or "go left" in message or "move west" in message:
            return ActionTypes.LEFT
        elif "move right" in message or "go right" in message or "move east" in message:
            return ActionTypes.RIGHT
        elif "move up" in message or "go up" in message or "move north" in message:
            return ActionTypes.UP
        elif "move down" in message or "go down" in message or "move south" in message:
            return ActionTypes.DOWN
        elif "stay" in message or "wait" in message or "don't move" in message:
            return ActionTypes.STAY
            
        # Default to STAY if no clear direction is found
        return ActionTypes.STAY

    def run(self, query: str, observation: np.ndarray) -> Dict[str, Any]:
        """Run the agent with the given query using LangGraph
        
        Args:
            query: The user's input query or instruction for the agent
            observation: The observation to act upon
            
        Returns:
            Dictionary containing the agent's response and any additional info
        """
        self.current_observation = observation

        try:        
            # Create a more informative prompt with current state information
            enhanced_query = f"""Current situation:
                - {self.current_observation}
                - Your current emotion is: {self.emotion.name if self.emotion else 'Not set'}

                User query: {query}

                Provide a helpful response:
            """
            
            # Use the LLM directly 
            logging.info(f"Invoking LangChain AgentExecutor: {enhanced_query}")
            
            # Get direct response from LLM
            response = self._agent_executor.invoke(enhanced_query)
            
            return {
                "action": response,
                "emotion": self._emotion.name if self.emotion else EmotionTypes.IDLE.name,
            }
        except Exception as e:
            logging.error(f"Error running agent: {str(e)}")
            return {"action": "An error occurred while processing your request."}