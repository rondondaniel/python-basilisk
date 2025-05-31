"""
Agent Brain - Core logic for decision-making and environment interaction
Using LangGraph to implement a ReAct agent pattern
"""
import numpy as np
import logging
import json
import re
from models import ActionTypes, EmotionTypes, CellTypes, AgentState
from typing import Dict, Any, List, Optional, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool, BaseTool
from langchain_ollama import OllamaLLM

# LangGraph imports
from langgraph.graph import StateGraph, END

class AgentBrain:
    def __init__(self):
        # Initialize with IDLE emotion by default
        self._emotion = EmotionTypes.IDLE
        self._emotions_memory: List[str] = ["IDLE"]  # Start with IDLE in history
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
            """Get the agent's current emotion (IDLE, HUNGRY, THIRSTY).
            Use this to understand your current needs - if HUNGRY, you should look for FOOD.
            If THIRSTY, you should look for WATER.
            """
            return f"Current emotion: {self._emotion.name if self._emotion else EmotionTypes.IDLE.name}"
            
        @tool
        def set_emotion(input_str: str) -> str:
            """Set the agent's emotion based on current needs.
            
            Args:
                input_str: JSON string with format: {"emotion": "emotion_name"}
                           where emotion_name is one of: IDLE, HUNGRY, THIRSTY
                           
            Example: {"emotion": "HUNGRY"} - Sets emotion to HUNGRY (you need FOOD)
                    {"emotion": "THIRSTY"} - Sets emotion to THIRSTY (you need WATER)
            """
            return self._emotion_tool(input_str)
            
        @tool
        def get_direction(input_str: str = "") -> str:
            """Get relative direction to a specified cell type.
            
            Args:
                input_str: JSON string with format: {"target": "cell_type"}
                           where cell_type is one of: food, water
                           
            Example: {"target": "food"} - Returns direction to nearest FOOD
                    {"target": "water"} - Returns direction to nearest WATER
            """
            try:
                input_data = json.loads(input_str) if input_str else {}
                target = input_data.get("target", "")
                
                if target.lower() == "food":
                    return self._get_relative_direction(CellTypes.FOOD)
                elif target.lower() == "water":
                    return self._get_relative_direction(CellTypes.WATER)
                else:
                    return "Please specify a valid target: food or water"
            except Exception as e:
                return f"Error parsing direction input: {str(e)}"
                
        @tool
        def describe_observation() -> str:
            """Get a description of the current environment observation.
            This gives you information about what's around you.
            """
            return self._get_state_description()
            
        @tool
        def retrieve_resource() -> str:
            """Retrieve a resource (FOOD or WATER) if the agent is adjacent to it.
            This tool should be used when you are next to a resource (food or water).
            - Use this when HUNGRY and next to FOOD
            - Use this when THIRSTY and next to WATER
            
            Returns:
                Success or failure message
            """
            if not self._current_observation is not None or not self._position:
                return "Error: No observation or position available"
                
            # Check if there's a resource adjacent to the agent
            row, col = self._position
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = row + dr, col + dc
                if 0 <= r < self._current_observation.shape[0] and 0 <= c < self._current_observation.shape[1]:
                    # Check if there's food or water at this position
                    cell_type = self._current_observation[r, c]
                    
                    if cell_type == CellTypes.FOOD and self._emotion == EmotionTypes.HUNGRY:
                        return "Success! You retrieved FOOD. Your hunger is satisfied."
                    elif cell_type == CellTypes.WATER and self._emotion == EmotionTypes.THIRSTY:
                        return "Success! You retrieved WATER. Your thirst is quenched."
                    elif cell_type == CellTypes.FOOD:
                        return "You found FOOD, but you're not HUNGRY right now."
                    elif cell_type == CellTypes.WATER:
                        return "You found WATER, but you're not THIRSTY right now."
            
            return "No resources found nearby. Move closer to a resource first."
                
        # Store tools for LangGraph
        self.tools = [
            get_emotion, 
            set_emotion, 
            get_direction, 
            describe_observation, 
            retrieve_resource
        ]
        
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

    def _update_emotion_based_on_previous_emotions(self, emotion_history):
        """Update the agent's emotion based on previous emotions and environmental factors
        
        Args:
            emotion_history: List of previous emotions
        """
        # Check if we have any food or water in observation
        has_food_nearby = False
        has_water_nearby = False
        food_dist = float('inf')
        water_dist = float('inf')
        
        # Only check if we have a valid observation and position
        if self._current_observation is not None and self._position is not None:
            # Get distances to food and water
            food_positions = np.where(self._current_observation == CellTypes.FOOD)
            water_positions = np.where(self._current_observation == CellTypes.WATER)
            
            if len(food_positions[0]) > 0:
                # Calculate Manhattan distance to nearest food
                for i in range(len(food_positions[0])):
                    fr, fc = food_positions[0][i], food_positions[1][i]
                    dist = abs(fr - self._position[0]) + abs(fc - self._position[1])
                    food_dist = min(food_dist, dist)
                has_food_nearby = food_dist <= 3  # Consider food "nearby" if within 3 cells
                
            if len(water_positions[0]) > 0:
                # Calculate Manhattan distance to nearest water
                for i in range(len(water_positions[0])):
                    wr, wc = water_positions[0][i], water_positions[1][i]
                    dist = abs(wr - self._position[0]) + abs(wc - self._position[1])
                    water_dist = min(water_dist, dist)
                has_water_nearby = water_dist <= 3  # Consider water "nearby" if within 3 cells
        
        # If no history or short history, initialize based on environment
        if not emotion_history or len(emotion_history) < 2:
            if has_food_nearby:
                self._emotion = EmotionTypes.HUNGRY
                print("\033[38;5;208m😋 EMOTION PREDICTION: HUNGRY (food nearby)\033[0m")
            elif has_water_nearby:
                self._emotion = EmotionTypes.THIRSTY
                print("\033[38;5;39m🥤 EMOTION PREDICTION: THIRSTY (water nearby)\033[0m")
            else:
                self._emotion = EmotionTypes.IDLE
                print("\033[38;5;250m😴 EMOTION PREDICTION: IDLE (no resources nearby)\033[0m")
            return
        
        # Get most recent emotions (last 3, excluding current one if it exists)
        recent_emotions = emotion_history[-3:] if len(emotion_history) >= 3 else emotion_history
        most_recent = recent_emotions[-1] if recent_emotions else None
        
        # Pattern-based emotion cycling with environmental influence
        if most_recent == "IDLE":
            # After being idle, prioritize nearby resources or default to hungry
            if has_food_nearby:
                self._emotion = EmotionTypes.HUNGRY
                print("\033[38;5;208m😋 EMOTION PREDICTION: HUNGRY (food nearby)\033[0m")
            elif has_water_nearby:
                self._emotion = EmotionTypes.THIRSTY
                print("\033[38;5;39m🥤 EMOTION PREDICTION: THIRSTY (water nearby)\033[0m")
            else:
                self._emotion = EmotionTypes.HUNGRY  # Default cycle: IDLE -> HUNGRY
                print("\033[38;5;208m😋 EMOTION PREDICTION: HUNGRY (cycle from IDLE)\033[0m")
                
        elif most_recent == "HUNGRY":
            # After being hungry, either stay hungry if food is nearby or switch to thirsty
            if has_food_nearby and food_dist < water_dist:
                # Stay hungry if food is closer than water
                self._emotion = EmotionTypes.HUNGRY
                print("\033[38;5;208m😋 EMOTION PREDICTION: HUNGRY (food is closer)\033[0m")
            else:
                self._emotion = EmotionTypes.THIRSTY  # Default cycle: HUNGRY -> THIRSTY
                print("\033[38;5;39m🥤 EMOTION PREDICTION: THIRSTY (cycle from HUNGRY)\033[0m")
                
        elif most_recent == "THIRSTY":
            # After being thirsty, either stay thirsty if water is nearby or switch to idle
            if has_water_nearby and water_dist < food_dist:
                # Stay thirsty if water is closer than food
                self._emotion = EmotionTypes.THIRSTY
                print("\033[38;5;39m🥤 EMOTION PREDICTION: THIRSTY (water is closer)\033[0m")
            else:
                self._emotion = EmotionTypes.IDLE  # Default cycle: THIRSTY -> IDLE
                print("\033[38;5;250m😴 EMOTION PREDICTION: IDLE (cycle from THIRSTY)\033[0m")
        else:
            # If we can't determine the pattern, default to IDLE
            self._emotion = EmotionTypes.IDLE
            print("\033[38;5;250m😴 EMOTION PREDICTION: IDLE (default)\033[0m")
        
    
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
                emotion_desc = f"Current emotion: {self._emotion.name if self._emotion else EmotionTypes.IDLE.name}"
                
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
            
            # Add important guidance based on current emotion
            emotion_guidance = ""
            if self._emotion == EmotionTypes.HUNGRY:
                emotion_guidance = "\nYou are HUNGRY! Your priority should be to find and retrieve FOOD. Use get_direction({\"target\": \"food\"}) to locate food and move towards it. When adjacent to food, use retrieve_resource() to consume it."
            elif self._emotion == EmotionTypes.THIRSTY:
                emotion_guidance = "\nYou are THIRSTY! Your priority should be to find and retrieve WATER. Use get_direction({\"target\": \"water\"}) to locate water and move towards it. When adjacent to water, use retrieve_resource() to consume it."
            else:
                emotion_guidance = "\nYou are IDLE! Your priority should be to find and retrieve FOOD or WATER. Use get_direction({\"target\": \"food\"}) to locate food and move towards it. When adjacent to food, use retrieve_resource() to consume it."
            
            # Add tool documentation with examples
            prompt += f"\n\n{emotion_guidance}\n\nYou have the following tools available:\n"
            
            # Add each tool with better formatting
            for tool in self.langchain_tools:
                prompt += f"- {tool.name}: {tool.description}\n"
            
            # Add clear instructions for tool usage
            prompt += """
            \nTo use a tool, respond with: [TOOL] tool_name(parameters) [/TOOL]\n
            Examples:\n
            [TOOL] get_emotion() [/TOOL] - Check your current emotion\n
            [TOOL] set_emotion({\"emotion\": \"HUNGRY\"}) [/TOOL] - Set your emotion to HUNGRY\n
            [TOOL] get_direction({\"target\": \"food\"}) [/TOOL] - Find direction to food\n
            [TOOL] retrieve_resource() [/TOOL] - Retrieve adjacent resource\n
            \nAfter gathering information, you must decide on a movement direction (UP, DOWN, LEFT, RIGHT, or STAY).\n"
            \nClearly state your movement intention with phrases like 'I will move UP', 'Going LEFT', etc.\n"
            """

            logging.info(f"\n\033[1m Agent Prompt: {prompt}\033[0m")
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
                
                # Extract action from LLM response
                action = self._extract_action_from_message(llm_response)
                
                # Print decision summary with visual indicators
                print("\033[1m🤖 Decision Summary:\033[0m")
                direction_emoji = {
                    ActionTypes.UP: "⬆️",
                    ActionTypes.DOWN: "⬇️",
                    ActionTypes.LEFT: "⬅️",
                    ActionTypes.RIGHT: "➡️",
                    ActionTypes.STAY: "⏹️"
                }.get(action, "❓")
                
                # Display direction with color
                print(f"\033[1;96m{direction_emoji} Direction: {action.name}\033[0m")
                
                # Extract reasoning using various patterns
                reasoning = ""
                reasoning_patterns = [
                    # Common reasoning phrases
                    ("because", "because"),
                    ("since", "since"),
                    ("as", "as"),
                    ("due to", "due to"),
                    ("so that", "so that"),
                    ("in order to", "in order to")
                ]
                
                # Try to find reasoning based on patterns
                for keyword, prefix in reasoning_patterns:
                    if keyword in llm_response.lower():
                        parts = llm_response.lower().split(keyword)
                        if len(parts) > 1:
                            # Extract the part after the keyword until end of sentence or line
                            reason_text = parts[1].split(".")[0].strip()
                            if len(reason_text) > 3:  # Only use if we got meaningful text
                                reasoning = f"{prefix} {reason_text}"
                                break
                
                # If no reasoning found with patterns, look for sentences with directional words
                if not reasoning:
                    direction_words = ["up", "down", "left", "right", "north", "south", "east", "west"]
                    for line in llm_response.split("\n"):
                        if any(word in line.lower() for word in direction_words):
                            # Take the shortest reasonable explanation
                            if 20 < len(line) < 100:
                                reasoning = line.strip()
                                break
                
                # If still no reasoning, use the last non-empty line as fallback
                if not reasoning:
                    lines = [line.strip() for line in llm_response.split("\n") if line.strip()]
                    if lines:
                        reasoning = lines[-1]
                
                # Truncate reasoning if too long
                if len(reasoning) > 80:
                    reasoning = reasoning[:77] + "..."
                
                # Display reasoning with color
                if reasoning:
                    print(f"\033[38;5;214m🧠 Reasoning: {reasoning}\033[0m")
                
                # Display current emotion with color and emoji
                if self._emotion:
                    emotion_emoji = {
                        "HUNGRY": "😋",
                        "THIRSTY": "🥤",
                        "IDLE": "😴"
                    }.get(self._emotion.name, "😐")
                    emotion_color = {
                        "HUNGRY": "\033[38;5;208m",  # Orange
                        "THIRSTY": "\033[38;5;39m",  # Blue
                        "IDLE": "\033[38;5;250m"     # Gray
                    }.get(self._emotion.name, "\033[38;5;7m")
                    print(f"{emotion_color}{emotion_emoji} Current emotion: {self._emotion.name}\033[0m")
                
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
                
                # Check if tool exists
                tool_found = False
                for tool_fn in self.tools:
                    if hasattr(tool_fn, '_tool_config') and (
                       tool_fn._tool_config.get('name') == tool_name or 
                       tool_fn.__name__ == tool_name):
                        tool_found = True
                        # Execute tool
                        try:
                            input_value = tool_args.get('input', '')
                            result = tool_fn(input_value) if input_value else tool_fn()
                            tool_result = f"Tool '{tool_name}' result: {str(result)}"
                            
                            # Print tool result with nice formatting
                            print(f"\033[38;5;35m💡 TOOL RESULT: '{tool_name}' → {str(result)}\033[0m")
                            
                            # Special handling for set_emotion - update emotion history
                            if tool_name == 'set_emotion' and self._emotion:
                                # Save emotion to history
                                self._emotions_memory.append(self._emotion.name)
                                print(f"\033[38;5;208m😊 EMOTION UPDATED: {self._emotion.name}\033[0m")
                            
                            tool_results.append(AIMessage(content=tool_result))
                        except Exception as e:
                            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                            print(f"\033[31m⚠️ TOOL ERROR: {error_msg}\033[0m")
                            tool_results.append(AIMessage(content=error_msg))
                        break
                
                if not tool_found:
                    error_msg = f"Unknown tool: '{tool_name}'. Available tools: {', '.join([t.__name__ for t in self.tools if hasattr(t, '__name__')])}" 
                    print(f"\033[31m⚠️ UNKNOWN TOOL: {tool_name}\033[0m")
                    tool_results.append(AIMessage(content=error_msg))
            
            # Add tool results to messages
            messages.extend(tool_results)
            return {"messages": messages}
        
        # 3. Emotion update node 
        def update_emotion(state: AgentState) -> AgentState:
            """Update the agent's emotion based on previous emotions"""
            # Always update emotion and ensure it's displayed
            previous_emotion = self._emotion.name if self._emotion else None
            
            # Update emotion based on current state
            self._update_emotion_based_on_previous_emotions(self._emotions_memory)
            current_emotion = self._emotion.name if self._emotion else EmotionTypes.IDLE.name
            
            # Always display the current emotion, even if unchanged
            emotion_emoji = {
                "HUNGRY": "😋",
                "THIRSTY": "🥤",
                "IDLE": "😴"
            }.get(current_emotion, "😐")
            emotion_color = {
                "HUNGRY": "\033[38;5;208m",  # Orange
                "THIRSTY": "\033[38;5;39m",  # Blue
                "IDLE": "\033[38;5;250m"     # Gray
            }.get(current_emotion, "\033[38;5;7m")
            
            # Only show a change message if the emotion actually changed
            if previous_emotion != current_emotion:
                print(f"{emotion_color}💫 EMOTION CHANGED: {previous_emotion} → {current_emotion} {emotion_emoji}\033[0m")
            else:
                print(f"{emotion_color}🔄 EMOTION MAINTAINED: {current_emotion} {emotion_emoji}\033[0m")
                
            # Update the agent state with the current emotion
            return {"emotion": current_emotion}
        
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
        
        # use_tools -> update_emotion (changed from agent to ensure emotion updates run)
        builder.add_edge("use_tools", "update_emotion")
        
        # Add edge from agent to update_emotion when there's a final decision
        # This ensures emotions are updated before action extraction
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
        
        # Add more comprehensive direction phrases
        # LEFT patterns
        if any(phrase in message for phrase in [
            "move left", "go left", "move west", "go west", 
            "i will move left", "moving left", "i'll move left",
            "i'll go left", "moving west", "i will go left",
            "i should move left", "i should go left", "heading left"]):
            print("\033[1;32m👉 DIRECTION DETECTED: LEFT\033[0m")
            return ActionTypes.LEFT
            
        # RIGHT patterns
        elif any(phrase in message for phrase in [
            "move right", "go right", "move east", "go east", 
            "i will move right", "moving right", "i'll move right",
            "i'll go right", "moving east", "i will go right",
            "i should move right", "i should go right", "heading right"]):
            print("\033[1;32m👈 DIRECTION DETECTED: RIGHT\033[0m")
            return ActionTypes.RIGHT
            
        # UP patterns
        elif any(phrase in message for phrase in [
            "move up", "go up", "move north", "go north", 
            "i will move up", "moving up", "i'll move up",
            "i'll go up", "moving north", "i will go up",
            "i should move up", "i should go up", "heading up", "heading north"]):
            print("\033[1;32m👆 DIRECTION DETECTED: UP\033[0m")
            return ActionTypes.UP
            
        # DOWN patterns
        elif any(phrase in message for phrase in [
            "move down", "go down", "move south", "go south", 
            "i will move down", "moving down", "i'll move down",
            "i'll go down", "moving south", "i will go down",
            "i should move down", "i should go down", "heading down", "heading south"]):
            print("\033[1;32m👇 DIRECTION DETECTED: DOWN\033[0m")
            return ActionTypes.DOWN
            
        # STAY patterns
        elif any(phrase in message for phrase in [
            "stay", "wait", "don't move", "do not move", "remain", 
            "stay put", "stay still", "hold position", "i'll stay",
            "i will stay", "staying still", "i will not move", "i won't move"]):
            print("\033[1;32m⏸️ DIRECTION DETECTED: STAY\033[0m")
            return ActionTypes.STAY
        
        # Search for single word directions (only if no specific phrases were found)
        direction_map = {
            'left': ActionTypes.LEFT,
            'west': ActionTypes.LEFT,
            'right': ActionTypes.RIGHT,
            'east': ActionTypes.RIGHT,
            'up': ActionTypes.UP,
            'north': ActionTypes.UP,
            'down': ActionTypes.DOWN,
            'south': ActionTypes.DOWN
        }
        
        # Check for standalone direction words
        for direction_word, action_type in direction_map.items():
            # Only match if it's a standalone word
            if re.search(r'\b' + direction_word + r'\b', message):
                print(f"\033[1;32m🔄 DIRECTION WORD DETECTED: {direction_word.upper()} -> {action_type.name}\033[0m")
                return action_type
            
        # Log that no direction was found
        print("\033[1;31m❓ NO DIRECTION DETECTED - DEFAULTING TO STAY\033[0m")
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