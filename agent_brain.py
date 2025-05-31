"""
Agent Brain - Core logic for decision-making and environment interaction
Using LangGraph to implement a ReAct agent pattern
"""
import numpy as np
import logging
import json
from models import ActionTypes, EmotionTypes, CellTypes, AgentState
from typing import Dict, Any, List, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool, BaseTool
from langchain_ollama import OllamaLLM


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
        
        # Initialize tools using the LangGraph tool decorator
        self._setup_tools()
        
        # Initialize the LangGraph
        self._setup_langgraph()
    
    @property
    def emotion(self):
        """Get the agent's current emotion"""
        return self._emotion
    
    @property
    def position(self):
        """Get the agent's current position"""
        return self._position
        
    @property
    def current_observation(self):
        """Get the agent's current observation"""
        return self._current_observation
        
    @current_observation.setter
    def current_observation(self, observation):
        """Set the agent's current observation"""
        self._current_observation = observation

    def _setup_tools(self):
        """Set up the tools for the LangGraph agent"""
        
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
        self.tools = [get_emotion, set_emotion, get_direction, describe_observation]
        
        # Create BaseTool objects directly for LangGraph
        self.langchain_tools = []
        for t in self.tools:
            if hasattr(t, '_tool_config'):
                self.langchain_tools.append(BaseTool(
                    name=t._tool_config['name'] or t.__name__,
                    description=t._tool_config['description'],
                    func=t
                ))

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
        
    
    def _setup_langgraph(self):
        """Set up the LangGraph for the ReAct agent"""
        # Create a new graph
        builder = StateGraph(AgentState)
        
        # Define nodes
        
        # 1. LLM agent node
        def agent(state: AgentState) -> AgentState:
            """Agent node that processes the messages and decides next steps"""
            # Prepare prompt with current state information
            messages = state["messages"]
            
            # Add environment info to the messages
            if self._current_observation is not None:
                observation_desc = self._get_state_description()
                emotion_desc = f"Current emotion: {self._emotion.name if self._emotion else 'None'}"
                
                # Check for resources in observation to add emoji indicators
                has_food = "food" in observation_desc.lower()
                has_water = "water" in observation_desc.lower()
                
                # Build message with resource indicators
                resource_indicators = ""
                if has_food:
                    resource_indicators += "🍎 Food detected! "
                if has_water:
                    resource_indicators += "💧 Water detected! "
                
                env_message = f"Environment observation: {observation_desc}\n{resource_indicators}\nEmotion: {emotion_desc}\n\nThink step by step about your situation and plan your next move carefully."
                
                # Add environment info as a system message if not already present
                if not any("Environment observation:" in str(m.content) for m in messages if hasattr(m, 'content')):
                    messages.append(HumanMessage(content=env_message))
            
            # Invoke LLM directly without binding tools
            # For Ollama LLM, we'll use a simpler approach for tools
            # Form a prompt with the messages and tool descriptions
            prompt = "\n".join([m.content for m in messages if hasattr(m, 'content')])
            prompt += "\n\nYou have the following tools available:\n"
            for tool in self.langchain_tools:
                prompt += f"- {tool.name}: {tool.description}\n"
            prompt += "\nTo use a tool, respond with: [TOOL] tool_name(parameters) [/TOOL]\n"
            
            # Invoke LLM with the prompt using streaming
            response_chunks = []
            try:
                import time
                import sys
                
                print("\n\033[1m🤔 Agent thinking...\033[0m")
                print("\033[90m", end="")  # Gray color for thinking
                
                # Stream response chunks
                for chunk in self._llm.stream(prompt):
                    # Print each chunk as it comes in
                    chunk_text = str(chunk)
                    sys.stdout.write(chunk_text)
                    sys.stdout.flush()
                    response_chunks.append(chunk_text)
                    # Small delay for readability (adjust as needed)
                    time.sleep(0.001)
                    
                print("\033[0m")  # Reset color
                
                # Combine chunks into full response
                llm_response = ''.join(response_chunks)
                
                # Extract decision from response for a summary
                action = self._extract_action_from_message(llm_response)
                action_name = action.name if action else "UNKNOWN"
                
                # Create a decision summary with color highlighting
                print("\033[1m🤖 Decision Summary:\033[0m")
                print(f"\033[96m➡️ Direction: {action_name}\033[0m")
                
                # Extract key phrases for decision reasoning (simplified)
                reasoning = ""
                if "because" in llm_response.lower():
                    reasoning = llm_response.lower().split("because")[1].split(".")[0]
                    print(f"\033[96m🧠 Reasoning: because{reasoning}\033[0m")
                
                # Log the response for debugging
                logging.debug(f"LLM Full Response: {llm_response}")
            except Exception as e:
                # Fallback if streaming not supported
                logging.info(f"Streaming error: {str(e)}. Using regular invoke.")
                llm_response = self._llm.invoke(prompt)
            
            # Create AI message from response
            result = AIMessage(content=llm_response)
            
            # Check if the response contains tool calls
            if "[TOOL]" in llm_response and "[/TOOL]" in llm_response:
                # Extract tool calls
                start_idx = llm_response.find("[TOOL]")
                end_idx = llm_response.find("[/TOOL]", start_idx)
                tool_call_text = llm_response[start_idx+6:end_idx].strip()
                
                # Parse tool call format: tool_name(parameters)
                if "(" in tool_call_text and ")" in tool_call_text:
                    tool_name = tool_call_text.split("(")[0].strip()
                    tool_args_str = tool_call_text.split("(")[1].split(")")[0].strip()
                    
                    # Print highlighted tool call
                    print(f"\033[93m⚡ TOOL CALL: {tool_name}({tool_args_str})\033[0m")
                    
                    # Add tool_calls to the message for our graph to process
                    # We're using a custom format that our use_tools node will understand
                    result.tool_calls = [{"name": tool_name, "args": {"input": tool_args_str}}]
            
            # Update messages
            messages.append(result)
            return {"messages": messages}
        
        # 2. Use tools node - for handling tool calls from the agent
        def use_tools(state: AgentState) -> AgentState:
            """Process any tool calls from the last message"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Check if there are tool calls
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
                
            # Process each tool call
            tool_results = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})
                
                # Find matching tool
                for tool_fn in self.tools:
                    if hasattr(tool_fn, '_tool_config') and tool_fn._tool_config.get('name') == tool_name:
                        # Execute tool
                        try:
                            input_value = tool_args.get('input', '')
                            result = tool_fn(input_value) if input_value else tool_fn()
                            tool_result = f"Tool '{tool_name}' result: {str(result)}"
                            
                            # Print tool result with nice formatting
                            print(f"\033[38;5;35m💡 TOOL RESULT: '{tool_name}' → {str(result)}\033[0m")
                            
                            tool_results.append(AIMessage(content=tool_result))
                        except Exception as e:
                            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                            tool_results.append(AIMessage(content=error_msg))
            
            # Add tool results to messages
            messages.extend(tool_results)
            return {"messages": messages}
        
        # 3. Emotion update node 
        def update_emotion(state: AgentState) -> AgentState:
            """Update the agent's emotion based on previous emotions"""
            if self._emotions_memory:
                self._update_emotion_based_on_previous_emotions(self._emotions_memory)
                
            return {"emotion": self._emotion.name if self._emotion else None}
        
        # 4. Action extraction node
        def extract_action(state: AgentState) -> Dict:
            """Extract an action from messages"""
            messages = state["messages"]
            
            # Look for action in the last AI message
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    action = self._extract_action_from_message(messages[i].content)
                    if action is not None:
                        return {"action": action.value}
                        
            # Default to STAY if no action found
            return {"action": ActionTypes.STAY.value}
            
        # Add nodes to the graph
        builder.add_node("agent", agent)
        builder.add_node("use_tools", use_tools)
        builder.add_node("update_emotion", update_emotion)
        builder.add_node("extract_action", extract_action) 
        
        # Define edges
        
        # Start with the agent
        builder.set_entry_point("agent")
        
        # Agent -> use_tools if tool call exists, otherwise -> update_emotion
        builder.add_conditional_edges(
            "agent",
            lambda state: "use_tools" if state["messages"] and hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls else "update_emotion"
        )
        
        # use_tools -> agent to continue reasoning
        builder.add_edge("use_tools", "agent")
        
        # Update emotion -> extract action
        builder.add_edge("update_emotion", "extract_action")
        
        # Extract action -> END
        builder.add_edge("extract_action", END)
        
        # Compile the graph
        self.graph = builder.compile()
        
    def _extract_action_from_message(self, message_content: str) -> Optional[ActionTypes]:
        """Extract an action from the agent's message
        
        Args:
            message_content: The content of the agent's message
            
        Returns:
            The action value (ActionTypes), or None if no action found
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
        try:
            # Update current observation and position
            self._current_observation = observation
            # Position would typically come from the environment
            # We'll assume it's in the center of any agent cell for now
            agent_pos = np.where(observation == CellTypes.AGENT)
            if len(agent_pos[0]) > 0:
                self._position = (agent_pos[0][0], agent_pos[1][0])
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "observation": observation,
                "position": self._position,
                "emotion": self._emotion.name if self._emotion else None,
                "action": None,
                "error": None
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Return the final state
            return {
                "action": final_state.get("action", ActionTypes.STAY.value),
                "emotion": final_state.get("emotion", EmotionTypes.IDLE.name),
                "messages": final_state.get("messages", []),
                "output": final_state["messages"][-1].content if final_state.get("messages") else ""
            }
        except Exception as e:
            logging.error(f"Error running agent with LangGraph: {str(e)}")
            return {
                "action": ActionTypes.STAY.value, 
                "error": str(e),
                "output": "An error occurred while processing your request."
            }