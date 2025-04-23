from enum import IntEnum

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