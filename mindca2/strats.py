from dtypes import Cell, ArrivalEvent, TerminationEvent, Event, IntLike
from typing import Type
import numpy.typing as npt
import numpy as np
from exploration_policies import Greedy, Boltzmann
from abc import ABC, abstractmethod
from grid import Grid, FixedGrid, GridArr
from logging import Logger
from eventgen import EventGen


from heapq import heappush, heappop


class Strat(ABC):
    """
    A call environment simulator plus a strategy/agent for handling
    call events, that is, to assign channels on incoming calls and
    possible reassigning channels on terminating calls.
    """
    grid: Grid

    def __init__(
        self,
        pp: dict,
        eventgen: EventGen,
        logger: Logger,
        sanity_check: bool = True,
    ):
        """
        :param sanity_check: Whether or not to verify that the channel reuse constraint
            is not broken each iteration of the simulation
        """
        self.rows = pp["rows"]
        self.cols = pp["cols"]
        self.n_channels = pp["n_channels"]
        self.n_events = pp["n_events"]
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
        action_ch = self.get_initial_action(event)

        # Discrete event simulation. Step through 'n_events' call events
        # and use the agent to act on each of them.
        for _event_i in range(self.n_events):
            gridarr = np.copy(self.grid.state)  # Copy before state is modified
            # Execute the given action (i.e. 'ch') for the given event on the grid.
            self.execute_action(event, action_ch)

            if self.sanity_check:
                # Check that the agent does not violate the reuse constraint.
                self.grid.validate_reuse_constr()

            # Generate the next event(s) and log some stats
            match event:
                case ArrivalEvent(t, cell):
                    n_incoming += 1
                    # Generate next incoming call
                    heappush(self.events, self.eventgen.arrival_event(t, cell))
                    if action_ch is None:
                        # No channel was available for assignment.
                        n_rejected += 1
                        self.logger.debug(f"Rejected call to {cell}")
                    else:
                        # Call accepted.
                        # Generate call duration for call and add END event
                        end_event = self.eventgen.termination_event(t, cell, action_ch)
                        heappush(self.events, end_event)
                case TerminationEvent(t, cell, _):
                    # TODO
                    pass

            # Prepare fo next simulation step.
            # Get the next event to handle,  and the corresponding action to handle it.
            next_event = heappop(self.events)
            next_action_ch = self.get_action(
                next_event,
                gridarr,
                cell,
                action_ch,
                type(event),
            )

            action_ch, event = next_action_ch, next_event
        self.logger.info(
            f"Rejected {n_rejected} of {n_incoming} ({n_rejected*100/n_incoming:.2f}%) calls"
        )

    @abstractmethod
    def get_action(self, event: Event, *_args, **_kwargs) -> int | None:
        """
        Action in response to event.

        For arrival events, the action is the channel to assign in the given cell.

        For termination events, the action is optionally the channel to reassign
        to the channel of th ending call. This is equivalent to ending the call
        on the channel given by the action, and letting the channel given by the
        event remain in use.
        """
        raise NotImplementedError()

    get_initial_action=get_action

    def execute_action(self, event: Event, ch: int | None) -> None:
        """
        Given an event and an action (i.e. 'ch'), execute the action on the grid.
        This amounts to marking a channel in use on arrival events and
        marking a channel as free on termination events.
        """
        if ch is not None:
            match event:
                case ArrivalEvent(t, cell):
                    self.grid.state[cell][ch] = 1
                case TerminationEvent(t, cell, _):
                    self.grid.state[cell][ch] = 0


class FAStrat(Strat):
    """
    Fixed assignment (FA) channel allocation.
    The set of channels is partitioned, and the partitions are permanently
    assigned to cells so that all cells can use all the channels assigned
    to them simultaneously without interference.
    """

    def __init__(self, pp,  *args, **kwargs):
        self.grid = FixedGrid(**pp)
        super().__init__(pp, *args, **kwargs)

    def get_action(self, event: Event, *args) -> int | None:
        """
        On arrival events, the first unused nominal channel is selected.

        No reassignment is done on termination events; thus the channel
        of the call is selected for termination.
        """
        match event:
            case ArrivalEvent(t, cell):
                # When a call arrives in a cell,
                # if any pre-assigned channel (i.e. a nominal channel) is unused;
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

    get_initial_action = get_action

class RLStrat(Strat):
    def __init__(self, pp: dict, *args, **kwargs):
        self.grid = Grid(**pp)
        super().__init__(pp, *args, **kwargs)
        self.exploration_policy = Greedy()
        self.losses = [0]
        self.gamma = pp['gamma']  # discount factor

    def get_initial_action(
        self,
        event: Event,
    ) -> int | None:
        """
        Return a channel to be assigned in response to 'cevent'.
        """
        # Choose A from S
        next_ch, next_max_ch = self.optimal_ch(event)
        return next_ch

    def get_action(
        self,
        next_event: Event,
        gridarr: GridArr,
        cell: Cell,
        ch: int | None,
        event_type: Type[Event]
    ) -> int | None:
        """Return a channel to be (re)assigned in response to 'next_cevent'.

        'cell', 'ch' and specify the action that  executed on
        'grid' in response to an event of type 'event_type'.
        """
        # Choose A' from S'
        next_ch, next_max_ch = self.optimal_ch(next_event)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        if (
            event_type is not TerminationEvent
            and ch is not None
            and next_ch is not None
        ):
            assert next_max_ch is not None
            # Observe reward from previous action, and
            # update q-values with one-step look-ahead
            self.update_qval(gridarr, cell, ch, next_event.cell, next_ch)
        return next_ch

    @abstractmethod
    def update_qval(
        self, gridarr:GridArr, cell: Cell, ch: int, next_cell: Cell, next_ch: int
    ):
        raise NotImplementedError()

    def optimal_ch(self, event: Event) -> tuple[int, int] | tuple[None, None]:
        """
        Select the channel fitting for assignment that
        that has the maximum q-value according to an exploration policy,
        or select the channel for termination that has the minimum
        q-value in a greedy fashion.

        Return (ch, max_ch) where 'ch' is the selected channel according to
        exploration policy and max_ch' is the greedy (still eligible) channel.
        'ch' (and 'max_ch') is None if no channel is eligible for assignment.
        """
        # (Number of) Channels in use at event cell
        inuse = np.nonzero(self.grid.state[event.cell])[0]
        n_used = len(inuse)

        # Should also include HOFF in the first branch if implemented
        if isinstance(event, ArrivalEvent):
            chs = self.grid.get_eligible_chs(event.cell)
            if len(chs) == 0:
                # No channels available for assignment,
                return None, None
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            assert n_used > 0

        qvals_dense = self.get_qvals(cell=event.cell, n_used=n_used, chs=chs)
        if isinstance(event, TerminationEvent):
            # Selecting a ch for reassigment is always greedy because no learning
            # is done on the reassignment actions.
            # Select minimum-valued channel
            amin_idx = np.argmin(qvals_dense)
            ch = max_ch = chs[amin_idx]
        else:
            ch, idx = self.exploration_policy.select_action(
                chs=chs, qvals=qvals_dense, time=event.time
            )
            amax_idx = np.argmax(qvals_dense)
            max_ch = chs[amax_idx]

        # If qvals blow up ('NaN's and 'inf's), ch becomes none.
        if ch is None:
            self.logger.error(f"ch is none for {event}\n{chs}\n{qvals_dense}\n")
            raise Exception
        self.logger.debug(f"Optimal ch: {ch} for event {event} of possibilities {chs}")
        return (ch, max_ch)

    @abstractmethod
    def get_qvals(
        self,
        cell: Cell,
        n_used: int,
        chs: IntLike | npt.NDArray[np.int_],
        *args,
        **kwargs,
    ):
        raise NotImplementedError()


class QTable(RLStrat):
    def __init__(self, pp, *args, **kwargs):
        super().__init__(pp, *args, **kwargs)
        # Learning rate for RL algorithm
        self.alpha = pp["alpha"]
        self.alpha_decay = pp["alpha_decay"]
        self.exploration_policy = Boltzmann(
            epsilon=pp["epsilon"], epsilon_decay=pp["epsilon_decay"]
        )
        self.qvals = np.zeros(self.grid.state.shape)

    @abstractmethod
    def feature_rep(self, cell: Cell, *args, **kwargs):
        """Feature representation of state"""
        raise NotImplementedError()

    def get_qvals(self, cell, n_used, chs, *args, **kwargs):
        """Get Q-Values for the given cell and all the given channels"""
        frep = self.feature_rep(cell, n_used)
        return self.qvals[frep][chs]

    def update_qval(
        self, gridarr:GridArr, cell: Cell, ch: int, next_cell: Cell, next_ch: int
    ):
        assert type(ch) == np.int64
        assert ch is not None
        if self.sanity_check:
            assert np.sum(gridarr != self.grid.state) == 1
        next_n_used = np.count_nonzero(self.grid.state[next_cell])
        next_qval = self.get_qvals(next_cell, next_n_used, next_ch)
        reward = self.grid.state.sum()
        target_q = reward + self.gamma * next_qval
        n_used = np.count_nonzero(gridarr[cell])
        q = self.get_qvals(cell=cell, n_used=n_used, chs=ch)
        td_err = target_q - q
        self.losses.append(td_err**2)
        frep = self.feature_rep(cell, n_used)
        self.qvals[frep][ch] += self.alpha * td_err
        self.alpha *= self.alpha_decay
        next_frep = self.feature_rep(next_cell, next_n_used)
        self.logger.debug(
            f"Q[{frep}][{ch}]:{q:.1f} -> {reward:.1f} + Q[{next_frep}][{next_ch}]:{next_qval:.1f}"
        )


class RS_SARSA(QTable):
    """
    Reduced-state SARSA.
    State consists of cell coordinates only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def feature_rep(self, cell: Cell, *args, **kwargs):
        return cell
