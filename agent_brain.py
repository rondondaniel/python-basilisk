"""
Agent Brain - Core logic for decision-making and environment interaction
"""
import numpy as np
import logging
import json
from models import ActionTypes, EmotionTypes, CellTypes
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from typing import Dict, Any
from env import BoardEnv

class AgentBrain:
    def __init__(self):
        self.state = None
        self.emotion = None
        self.position = None  # Store agent position
        
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
                name="observe_tool",
                func=self._observe_tool,
                description="Observes the environment around the agent. No input required."
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
        
        # Initialize the agent executor
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Initialize or reinitialize the agent executor with current settings"""
        try:
            # Use ZERO_SHOT_REACT_DESCRIPTION which is more compatible with multiple tools
            agent_executor = initialize_agent(
                tools=self._tools_list,
                llm=self._llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_tool_error=True,
                max_iterations=10
            )
            
            self._agent_executor = agent_executor
            logging.info("Successfully initialized agent with ZERO_SHOT_REACT_DESCRIPTION")
            
        except Exception as e:
            logging.error(f"Failed to initialize agent: {str(e)}")
            # Fall back to a more basic implementation
            try:
                # Import directly for the most up-to-date implementations
                from langchain.agents import create_openai_functions_agent
                from langchain.agents import AgentExecutor
                from langchain.schema.runnable import RunnablePassthrough
                from langchain.prompts import ChatPromptTemplate
                
                # Create a simple system prompt that doesn't rely on specific variables
                system_message = (
                    "You are an intelligent agent in a 2D grid world. "
                    "You can observe your surroundings, move in different directions, "
                    "and express different emotions. "
                    "Use the available tools to help answer the user's questions."
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_message),
                    ("user", "{input}"),
                    ("user", "Think through this step-by-step:"),
                ])
                
                # Create a simple chain directly
                chain = prompt | self._llm
                self._agent_executor = {
                    "input": lambda x: x,
                    "output": lambda x: {"output": x}
                }
                logging.info("Fallback to simple LLM chain without tools")
                
            except Exception as e2:
                logging.error(f"Failed second attempt to initialize agent: {str(e2)}")
                raise e
    
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

    def _move_tool(self, input_str: str, env=None) -> str:
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
            
    def _observe_tool(self, input_str: str = "", env=None) -> str:
        """Tool that allows the agent to observe its environment
        
        Args:
            input_str: Not used, but required by LangChain tool interface
            env: Optional environment to observe
            
        Returns:
            String describing what the agent observes in its environment
        """
        try:
            if env is None:
                return "No environment provided for observation."
                
            # Update state from the provided environment
            self.state = env.state
            self.position = env.agent_position
            
            # Create a description of what the agent sees
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
            
        except Exception as e:
            logging.error(f"Error in observe_tool: {str(e)}")
            return f"Error observing environment: {str(e)}"
    
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

    def init_state_from_env(self, env: BoardEnv) -> None:
        """Initialize agent state from an environment
        
        Args:
            env: BoardEnv instance to initialize state from
        """
        # Get state from environment without storing the environment itself
        self.state = env.reset()
        self.position = env.agent_position
        logging.info(f"Agent state initialized from environment. Position: {self.position}")
    
    def _get_state_description(self) -> str:
        """Get a description of the current state without needing an environment"""
        if self.state is None or self.position is None:
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

    def run(self, query: str, env: Optional[BoardEnv] = None) -> Dict[str, Any]:
        """Run the agent with the given query
        
        Args:
            query: The user's input query or instruction for the agent
            env: Optional environment to use for the interaction
            
        Returns:
            Dictionary containing the agent's response and any additional info
        """
        try:
            # Get observation from the provided environment
            if env is not None:
                # Update state from environment
                self.state = env.state
                self.position = env.agent_position
                observation = self._observe_tool("", env)
            else:
                observation = "No environment provided for observation." if self.state is None else self._get_state_description()
            
            # Create a more informative prompt with current state information
            enhanced_query = f"""Current situation:
- {observation}
- Your current emotion is: {self.emotion.name if self.emotion else 'Not set'}

User query: {query}

Provide a helpful response:"""
            
            # Use the LLM directly 
            logging.info(f"Running LLM with query: {enhanced_query}")
            
            # Get direct response from LLM
            response = self._llm.invoke(enhanced_query)
            
            # Update emotion based on the current state if environment is provided
            if env is not None:
                self._update_emotion_based_on_state(env)
            
            return {
                "output": response,
                "position": self.position,
                "emotion": self.emotion.name if self.emotion else None,
                "observation": observation,
                "state": self.state.tolist() if self.state is not None else None
            }
        except Exception as e:
            logging.error(f"Error running agent: {str(e)}")
            return {"error": str(e), "output": "An error occurred while processing your request."}
    
    def _update_emotion_based_on_state(self, env=None) -> None:
        """Update the agent's emotion based on the current environment state
        
        Args:
            env: Optional environment to use for updating emotion
        """
        # Use the provided environment or fall back to current state
        state = self.state
        position = self.position
        
        if env is not None:
            state = env.state
            position = env.agent_position
            
        if state is None or position is None:
            return
            
        # Get positions of food and water
        food_positions = np.argwhere(state == CellTypes.FOOD)
        water_positions = np.argwhere(state == CellTypes.WATER)
        
        # Calculate distances to food and water
        food_distance = float('inf')
        water_distance = float('inf')
        
        if len(food_positions) > 0:
            food_pos = tuple(food_positions[0])
            food_distance = abs(food_pos[0] - position[0]) + abs(food_pos[1] - position[1])
            
        if len(water_positions) > 0:
            water_pos = tuple(water_positions[0])
            water_distance = abs(water_pos[0] - position[0]) + abs(water_pos[1] - position[1])
        
        # Set emotion based on distances
        if food_distance <= water_distance and food_distance < 3:
            self.emotion = EmotionTypes.HUNGRY
        elif water_distance < food_distance and water_distance < 3:
            self.emotion = EmotionTypes.THIRSTY
        else:
            self.emotion = EmotionTypes.IDLE
