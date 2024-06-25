from typing import NamedTuple
import numpy as np

type IntLike = int | np.int_ | np.int64


class Cell(NamedTuple):
    row: int
    col: int

class ArrivalEvent(NamedTuple):
    time: float
    cell: Cell

class TerminationEvent(NamedTuple):
    time: float
    cell: Cell
    ch: int

type Event = ArrivalEvent | TerminationEvent

# event1 = ArrivalEvent(0.0, Cell(1,3))
# event2 = TerminationEvent(0.2, Cell(1,4), 2)

# match event1:
#     case ArrivalEvent(t, cell):
#         print("ArrivalEvent", t, cell)
#     case TerminationEvent(t, cell, ch):
#         print("TerminationEvent", t, cell, ch)
