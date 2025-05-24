import numpy as np
from enum import IntEnum
import logging
import json
from models import ActionTypes, EmotionTypes
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, initialize_agent
from langchain_ollama import OllamaLLM
from typing import Dict, Any, List


logging.getLogger().setLevel(logging.INFO)

class AgentBrain:
    def __init__(self, current_position: tuple, current_emotion: EmotionTypes):
        self._position = current_position
        self._emotion = current_emotion
        
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
                description="Returns the agent's current emotion. No input required."
            ),
            Tool(
                name="move_tool",
                func=self._move_tool,
                description="Moves the agent in a specified direction. Input should be a JSON string with format: {\"direction\": \"up|down|left|right\"}"
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
            self._agent_executor = AgentExecutor.from_agent_and_tools(
                agent=initialize_agent(
                    tools=self._tools_list,
                    llm=self._llm,
                    verbose=True
                ),
                tools=self._tools_list,
                verbose=True,
                handle_tool_error=True,
                max_iterations=5,
                max_execution_time=60,  # Increased from 2 seconds which was too short
            )
        except Exception as e:
            logging.error(f"Failed to initialize agent: {str(e)}")
            raise
    

    def _emotion_tool(self, input_str: str = "") -> str:
        """Tool that returns the agent's a new emotion based on passed parameters"""
        pass

    def _move_tool(self, input_str: str) -> str:
        """Tool that returns the agent's move decision based on ActionTypes model of the action space"""
        pass

    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent with the given query"""
        try:
            response = self._agent_executor.invoke({"input": query})
            return response
        except Exception as e:
            logging.error(f"Error running agent: {str(e)}")
            return {"error": str(e), "output": "An error occurred while processing your request."}
        

class Agent:
    def __init__(self):
        self.agent_brain = AgentBrain()

    @property
    def position(self):
        return self.agent_brain.position

    @position.setter
    def position(self, value):
        self.agent_brain.position = value

    @property
    def emotion(self):
        return self.agent_brain.emotion

    @emotion.setter
    def emotion(self, value):
        self.agent_brain.emotion = value

    def take_action(self, state, action_space):
        # set agent position
        self.position = state
        logging.info(f"Agent emotion: {self.emotion.name}")
        return np.random.choice(action_space)