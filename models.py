from enum import IntEnum
from typing import List, Tuple, TypedDict, Union, Optional
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
class ActionTypes(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4

class CellTypes(IntEnum):
    EMPTY = 0
    FOOD = 1
    WATER = 2
    AGENT = 3

class EmotionTypes(IntEnum):
    IDLE = 0
    HUNGRY = 1
    THIRSTY = 2

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    observation: np.ndarray
    position: Optional[Tuple[int, int]]
    emotion: Optional[str]
    action: Optional[int]
    error: Optional[str]