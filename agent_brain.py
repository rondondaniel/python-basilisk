"""
Agent Brain - Core logic for decision-making and environment interaction
"""
import numpy as np
import logging
import json
from models import ActionTypes, EmotionTypes, CellTypes
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from typing import Dict, Any, List

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
        
        # Initialize tools
        self._tools_list = [
            Tool(
                name="emotion_tool",
                func=self._emotion_tool,
                description="Returns or sets the agent's emotion. Input (optional): {\"emotion\": \"idle|hungry|thirsty\"}"
            ),
            Tool(
                name="move_tool",
                func=self._move_tool,
                description="Moves the agent in a specified direction. Input: {\"direction\": \"left|right|up|down|stay\"}"
            ),
            Tool(
                name="describe_observation_tool",
                func=self._describe_observation_tool,
                description="Observes the environment around the agent. Input: {\"observation\": \"observation\"}"
            ),
        ]
        
        # Initialize prompt template
        self._prompt_template = PromptTemplate.from_template(
            """
            You are an intelligent agent that can move in a 2D grid and express emotions.
            
            Available tools:
            {tools}
            
            TO USE A TOOL, YOU MUST USE THIS EXACT FORMAT:
            Thought: I need to use a tool to help answer the question.
            Action: tool_name
            Action Input: the input to the tool (may be empty for some tools)
            
            Example using the emotion_tool:
            Thought: I want to check my current emotion.
            Action: emotion_tool
            Action Input: 
            
            Example using the move_tool:
            Thought: I want to move up.
            Action: move_tool
            Action Input: {"direction": "up"}
            
            Question: {input}
            {agent_scratchpad}
            """
        )
        
        self.agent_executor = None

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
            # Input validation
            if not input_str:
                return "Error: Direction not specified. Please provide a direction (left, right, up, down, stay)."
            
            # Parse the input JSON
            input_data = json.loads(input_str) if isinstance(input_str, str) else input_str
            direction_str = input_data.get("direction", "").upper()
            
            # Map direction string to ActionTypes
            try:
                action = ActionTypes[direction_str]
            except KeyError:
                valid_directions = [a.name.lower() for a in ActionTypes]
                return f"Invalid direction: {direction_str.lower()}. Valid directions are: {', '.join(valid_directions)}"
            
            # Update state based on the action if environment is provided
            if env is not None:
                try:
                    # Call environment's step method with the action
                    new_state, reward, done, info = env.step(action)
                    self.state = new_state
                    self.position = env.agent_position
                    
                    logging.info(f"Agent took action: {action.name}, moved to {self.position}")
                    return f"Action taken: {action.name}. New position: {self.position}. Reward: {reward}. Done: {done}"
                except Exception as e:
                    logging.error(f"Environment step error: {str(e)}")
                    return f"Error during environment step: {str(e)}"
            else:
                return f"Action {action.name} recorded, but no environment was provided."
                
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
        if self._state is None or self._position is None:
            return "No state information available."
        
        # Create a description based on the stored state
        food_positions = np.argwhere(self.state == CellTypes.FOOD)
        water_positions = np.argwhere(self.state == CellTypes.WATER)
        
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

    def _update_emotion_based_on_previous_emotions(self, emotions_memory: List[str]) -> None:
        """Update the agent's emotion based on previous emotions
        
        Args:
            emotions_memory: List of previous emotions
        """
        pass
        
    
    def run(self, query: str, observation: np.ndarray) -> Dict[str, Any]:
        """Run the agent with the given query
        
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
            response = self.agent_executor.invoke(enhanced_query)
            
            return {
                "action": response,
                "emotion": self._emotion.name if self.emotion else EmotionTypes.IDLE.name,
            }
        except Exception as e:
            logging.error(f"Error running agent: {str(e)}")
            return {"action": "An error occurred while processing your request."}