from enum import Enum


class Periodicity (Enum):
    DAILY = 1
    PERMIN = 2
    PERSECOND = 3
    TICK = 4


class Side (Enum):
    BUY = 1
    SELL = 2
    SHORT = 3
    COVER = 4
    NONE = 5


class Type (Enum):
    IND = 1
    STK = 2
    FUT = 3
    OPT = 4


class Right (Enum):
    CALL = 1
    PUT = 2


class Stoptype (Enum):
    NONE = 1
    POINTS = 2
    PRICE = 3
    TRAILING = 4
