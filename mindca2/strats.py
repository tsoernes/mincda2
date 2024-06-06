from dtypes import Cell, ArrivalEvent, TerminationEvent


from heapq import heappush, heappop


class Strat:
    """
    A call environment simulator plus a strategy/agent for handling
    call events, that is, to assign channels on incoming calls and
    possible reassigning channels on terminating calls.
    """
    def __init__(self, pp, eventgen, grid, logger,
                 sanity_check=True):
        self.rows = pp['rows']
        self.cols = pp['cols']
        self.n_channels = pp['n_channels']
        self.n_events = pp['n_events']
        self.grid = grid
        # A heap queue of sorted (time, call_event) pairs.
        # The queue is sorted such that the call events with the lowest event
        # time is popped first.
        self.events = []  # Call events
        self.eventgen = eventgen
        self.sanity_check = sanity_check
        self.logger = logger

    def simulate(self) -> None:
        """
        Run caller environment simulation.

        Runs through simulation event by event and uses the agent to
        handle the events.
        """
        # Generate initial call events; one for each cell.
        # Each initial call event must be a an arriving call.
        for r in range(self.rows):
            for c in range(self.cols):
                heappush(self.events, self.eventgen.arrival_event(0, Cell(r, c)))

        # Count number of incoming calls and number of rejected calls
        n_incoming = 0
        n_rejected = 0
        # Get the first call event and a responding action to handle it
        event = heappop(self.events)
        ch = self.get_action(event)

        # Discrete event simulation. Step through 'n_events' call events
        # and use the agent to act on each of them.
        for _event_i in range(self.n_events):

            # Execute the given action (i.e. 'ch') for the given event on the grid.
            t, cell = event.time, event.cell
            self.execute_action(event, ch)

            # Check that the agent does not violate the reuse constraint.
            if self.sanity_check and not self.grid.validate_reuse_constr():
                # If this happens, the agent has performed an illegal action.
                # This should never happend and indicated a bug in the agent.
                self.logger.error(f"Reuse constraint broken: {self.grid}")
                raise Exception

            match event:
                case ArrivalEvent(t, cell):
                    n_incoming += 1
                    # Generate next incoming call
                    heappush(self.events, self.eventgen.arrival_event(t, cell))
                    if ch is None:
                        # No channel was available for assignment.
                        n_rejected += 1
                        self.logger.debug(f"Rejected call to {cell}")
                    else:
                        # Call accepted.
                        # Generate call duration for call and add END event
                        end_event = self.eventgen.termination_event(t, cell, ch)
                        heappush(self.events, end_event)
                case TerminationEvent(t, cell, ch):
                    pass

            # Prepare fo next simulation step.
            # Get the next event to handle,  and the corresponding action to handle it.
            event = heappop(self.events)
            next_ch = self.get_action( event)
            ch, event = next_ch,  event
        self.logger.info(f"Rejected {n_rejected} of {n_incoming} calls")

    def get_action(self, event, *_args, **_kwargs) -> int | None:
        """
        Action in response to event.

        For arrival events, the action is the channel to assign in the given cell.

        For termination events, the action is optionally the channel to reassign
        to the channel of th ending call. This is equivalent to ending the call
        on the channel given by the action, and letting the channel given by the
        event remain in use.
        """
        raise NotImplementedError()

    def execute_action(self, *_args, **_kwrags):
        """
        Change which channels are marked is free or in use on the grid
        """
        raise NotImplementedError()


class FAStrat(Strat):
    """
    Fixed assignment (FA) channel allocation.
    The set of channels is partitioned, and the partitions are permanently
    assigned to cells so that all cells can use all the channels assigned
    to them simultaneously without interference.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action(self, event, *args) -> int | None:
        match event:
            case ArrivalEvent(t, cell):
                # When a call arrives in a cell,
                # if any pre-assigned channel is unused;
                # it is assigned, else the call is blocked.
                ch = None
                for idx, isNom in enumerate(self.grid.nom_chs[cell]):
                    if isNom and self.grid.state[cell][idx] == 0:
                        ch = idx
                        break
                return ch
            case TerminationEvent(t, cell, ch):
                # No rearrangement is done when a call terminates.
                return ch

    def execute_action(self, event, ch) -> None:
        if ch is not None:
            match event:
                case ArrivalEvent(t, cell):
                    self.grid.state[cell][ch] = 1
                case TerminationEvent(t, cell, _):
                    self.grid.state[cell][ch] = 0
