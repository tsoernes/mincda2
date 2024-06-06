import numpy as np
from dtypes import Cell, ArrivalEvent, TerminationEvent


class EventGen:
    """
    Event generator
    """

    def __init__(self, rows, cols, call_rate, call_duration, *args, **kwargs):
        self.rows = rows
        self.cols = cols
        # Avg. time between arriving calls
        self.call_intertimes = 1 / call_rate
        self.call_duration = call_duration

    def arrival_event(self, t: float, cell: Cell) -> ArrivalEvent:
        """
        Generate a call arrival event indicating an arriving call at the given cell at an
        exponentially distributed time dt from t.

        """
        dt = np.random.exponential(self.call_intertimes)
        return ArrivalEvent(dt + t, cell)

    def termination_event(self, t: float, cell: Cell, ch: int) -> TerminationEvent:
        """
        Generate END event for a call

        Return tuple with
        - Event time
        - Type of call event (an END event)
        - Cell number to end a call in
        - Call channel to end
        """
        call_duration = np.random.exponential(self.call_duration)
        end_time = call_duration + t
        return TerminationEvent(end_time, cell, ch)
