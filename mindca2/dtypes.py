from collections import namedtuple
from dataclasses import dataclass

# from enum import Enum, auto
Cell = namedtuple("Cell", ["row", "col"])

# class EventType(Enum):
#     """
#     Types of call events
#     """

#     NEW = auto()  # Incoming call
#     END = auto()  # End a current call

# @dataclass
# class ArrivalEvent:
#     """
#     A call arrival event indicating an arriving call at the given cell and time
#     """
#     time: float
#     cell: Cell

ArrivalEvent = namedtuple("ArrivalEvent", ["time", "cell"])

# @dataclass
# class TerminationEvent:
#     """
#     - Event time
#     - Cell number to end a call in
#     - The channel of the call to end
#     """
#     time: float
#     cell: Cell
#     ch: int

TerminationEvent = namedtuple("TerminationEvent", ["time", "cell", "ch"])

Event = ArrivalEvent | TerminationEvent

event1 = ArrivalEvent(0.0, Cell(1,3))
event2 = TerminationEvent(0.2, Cell(1,4), 2)

# match event1:
#     case ArrivalEvent(t, cell):
#         print("ArrivalEvent", t, cell)
#     case TerminationEvent(t, cell, ch):
#         print("TerminationEvent", t, cell, ch)
