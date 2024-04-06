"""
main class for building a DL pipeline.

"""
from enum import Enum, auto


class Stage(Enum):
    TRAIN = auto()
    DEV = auto()
    TEST = auto()

