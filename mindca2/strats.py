from dtypes import Cell, ArrivalEvent, TerminationEvent, Event
import numpy as np
from exploration_policies import Greedy, Boltzmann
from abc import ABC, abstractmethod
from grid import Grid


from heapq import heappush, heappop


class Strat(ABC):
    """
    A call environment simulator plus a strategy/agent for handling
    call events, that is, to assign channels on incoming calls and
    possible reassigning channels on terminating calls.
    """

    def __init__(self, pp, eventgen, grid, logger, sanity_check: bool = True):
        """
        :param sanity_check: Whether or not to verify that the channel reuse constraint
            is not broken each iteration of the simulation
        """
        self.rows = pp["rows"]
        self.cols = pp["cols"]
        self.n_channels = pp["n_channels"]
        self.n_events = pp["n_events"]
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
        action_ch = self.get_action(event)

        # Discrete event simulation. Step through 'n_events' call events
        # and use the agent to act on each of them.
        for _event_i in range(self.n_events):
            # Execute the given action (i.e. 'ch') for the given event on the grid.
            self.execute_action(event, action_ch)

            # Check that the agent does not violate the reuse constraint.
            if self.sanity_check and not self.grid.validate_reuse_constr():
                # If this happens, the agent has performed an illegal action.
                # This should never happend and indicated a bug in the agent.
                self.logger.error(f"Reuse constraint broken: {self.grid}")
                raise Exception

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
            next_action_ch = self.get_action(next_event)
            action_ch, event = next_action_ch, next_event
        self.logger.info(f"Rejected {n_rejected} of {n_incoming} calls")

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

    @abstractmethod
    def execute_action(self, *_args, **_kwrags) -> None:
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

    def get_action(self, event: Event, *args) -> int | None:
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

    def execute_action(self, event: Event, ch: int | None) -> None:
        if ch is not None:
            match event:
                case ArrivalEvent(t, cell):
                    self.grid.state[cell][ch] = 1
                case TerminationEvent(t, cell, _):
                    self.grid.state[cell][ch] = 0


class RLStrat(Strat):
    def __init__(self, pp, *args, **kwargs):
        super().__init__(pp, *args, **kwargs)
        self.epsilon = self.epsilon0 = pp["epsilon"]
        self.exploration_policy = Greedy
        self.eps_log_decay = self.pp["eps_log_decay"]

        self.epsilon_decay = pp["epsilon_decay"]
        self.losses = [0]

    def get_init_action(self, event: Event) -> int:
        ch, _idx = self.optimal_ch(ce_type=event.event_type, cell=event.cell)
        return ch

    def get_action(self, next_cevent, grid, cell, ch, reward, ce_type, discount) -> int:
        next_ce_type, next_cell = next_cevent[1:3]
        # Choose A' from S'
        next_ch, next_max_ch = self.optimal_ch(next_ce_type, next_cell)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        if ce_type != CEvent.END and ch is not None and next_ch is not None:
            assert next_max_ch is not None
            # Observe reward from previous action, and
            # update q-values with one-step look-ahead
            self.update_qval(
                grid, cell, ch, reward, next_cell, next_ch, next_max_ch, discount
            )
        return next_ch

    def get_action2(
        self,
        next_event: Event,
        grid: Grid,
        cell: Cell,
        ch: int | None,
        reward: int,
        discount: float,
    ) -> int:
        # Choose A' from S'
        next_ch, next_max_ch = self.optimal_ch(next_event)
        # If there's no action to take, or no action was taken,
        # don't update q-value at all
        if (
            not isinstance(next_event, TerminationEvent)
            and ch is not None
            and next_ch is not None
        ):
            assert next_max_ch is not None
            # Observe reward from previous action, and
            # update q-values with one-step look-ahead
            self.update_qval(
                grid, cell, ch, reward, next_event.cell, next_ch, next_max_ch, discount
            )
        return next_ch

    def optimal_ch(self, ce_type, cell) -> tuple[int, float]:
        """
        Select the channel fitting for assignment that
        that has the maximum q-value according to an exploration policy,
        or select the channel for termination that has the minimum
        q-value in a greedy fashion.

        Return (ch, max_ch) where 'ch' is the selected channel according to
        exploration policy and max_ch' is the greedy (still eligible) channel.
        'ch' (and 'max_ch') is None if no channel is eligible for assignment.
        """
        # Channels in use at cell
        inuse = np.nonzero(self.grid[cell])[0]
        n_used = len(inuse)

        if ce_type == CEvent.NEW or ce_type == CEvent.HOFF:
            chs = NGF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                # No channels available for assignment,
                return (None, None, 0)
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            # or no channels in use to reassign
            assert n_used > 0

        # TODO If 'max_ch' turns out not to be useful, then don't return it and
        # avoid running a forward pass through the net if a random action is selected.
        qvals_dense = self.get_qvals(cell=cell, n_used=n_used, ce_type=ce_type, chs=chs)
        # Selecting a ch for reassigment is always greedy because no learning
        # is done on the reassignment actions.
        if ce_type == CEvent.END:
            amin_idx = np.argmin(qvals_dense)
            ch = max_ch = chs[amin_idx]
            p = 1
        else:
            ch, idx, p = self.exploration_policy.select_action(
                self.epsilon, chs, qvals_dense, cell
            )
            if self.eps_log_decay:
                self.epsilon = self.epsilon0 / np.sqrt(self.t * 60 / self.eps_log_decay)
            else:
                self.epsilon *= self.epsilon_decay
            amax_idx = np.argmax(qvals_dense)
            max_ch = chs[amax_idx]

        # If qvals blow up ('NaN's and 'inf's), ch becomes none.
        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
            raise Exception
        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}"
        )
        return (ch, max_ch, p)

    def optimal_ch2(self, event: Event, cell: Cell) -> tuple[int, float]:
        """
        Select the channel fitting for assignment that
        that has the maximum q-value according to an exploration policy,
        or select the channel for termination that has the minimum
        q-value in a greedy fashion.

        Return (ch, max_ch) where 'ch' is the selected channel according to
        exploration policy and max_ch' is the greedy (still eligible) channel.
        'ch' (and 'max_ch') is None if no channel is eligible for assignment.
        """
        # Channels in use at cell
        inuse = np.nonzero(self.grid[cell])[0]
        n_used = len(inuse)

        # Should also include HOFF in the first branch if implemented
        if isinstance(event, ArrivalEvent):
            chs = NGF.get_eligible_chs(self.grid, cell)
            if len(chs) == 0:
                # No channels available for assignment,
                return (None, None, 0)
        else:
            # Channels in use at cell, including channel scheduled
            # for termination. The latter is included because it might
            # be the least valueable channel, in which case no
            # reassignment is done on call termination.
            chs = inuse
            # or no channels in use to reassign
            assert n_used > 0

        # TODO If 'max_ch' turns out not to be useful, then don't return it and
        # avoid running a forward pass through the net if a random action is selected.
        qvals_dense = self.get_qvals(cell=cell, n_used=n_used, ce_type=ce_type, chs=chs)
        # Selecting a ch for reassigment is always greedy because no learning
        # is done on the reassignment actions.
        if ce_type == CEvent.END:
            amin_idx = np.argmin(qvals_dense)
            ch = max_ch = chs[amin_idx]
            p = 1
        else:
            ch, idx, p = self.exploration_policy.select_action(
                self.epsilon, chs, qvals_dense, cell
            )
            if self.eps_log_decay:
                self.epsilon = self.epsilon0 / np.sqrt(self.t * 60 / self.eps_log_decay)
            else:
                self.epsilon *= self.epsilon_decay
            amax_idx = np.argmax(qvals_dense)
            max_ch = chs[amax_idx]

        # If qvals blow up ('NaN's and 'inf's), ch becomes none.
        if ch is None:
            self.logger.error(f"ch is none for {ce_type}\n{chs}\n{qvals_dense}\n")
            raise Exception
        self.logger.debug(
            f"Optimal ch: {ch} for event {ce_type} of possibilities {chs}"
        )
        return (ch, max_ch, p)


class QTable(RLStrat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.pp["alpha"]
        self.alpha_decay = self.pp["alpha_decay"]
        self.lmbda = self.pp["lambda"]
        if self.lmbda is not None:
            self.logger.error("Using lambda returns")
        if self.pp["target"] != "discount":
            raise NotImplementedError(self.pp["target"])
        self.eps_log_decay = self.pp["eps_log_decay"]

    def get_qvals(self, cell, n_used, chs=None, *args, **kwargs):
        rep = self.feature_rep(cell, n_used)
        if chs is None:
            return self.qvals[rep]
        else:
            return self.qvals[rep][chs]

    def update_qval(
        self, grid, cell, ch, reward, next_cell, next_ch, next_max_ch, discount
    ):
        assert type(ch) == np.int64
        assert ch is not None
        if self.pp["verify_grid"]:
            assert np.sum(grid != self.grid) == 1
        next_n_used = np.count_nonzero(self.grid[next_cell])
        next_qval = self.get_qvals(next_cell, next_n_used, next_ch)
        target_q = reward + discount * next_qval
        # Counting n_used of self.grid instead of grid yields significantly lower
        # blockprob on (TT-)SARSA for unknown reasons.
        n_used = np.count_nonzero(grid[cell])
        q = self.get_qvals(cell, n_used, ch)
        td_err = target_q - q
        self.losses.append(td_err**2)
        frep = self.feature_rep(cell, n_used)
        if self.lmbda is None:
            self.qvals[frep][ch] += self.alpha * td_err
        else:
            self.el_traces[frep][ch] += 1
            self.qvals += self.alpha * td_err * self.el_traces
            self.el_traces *= discount * self.lmbda
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
        self.qvals = np.zeros(self.dims)

    def feature_rep(self, cell, n_used):
        return cell


class RS_SARSA_Strat(Strat):
    """
    RS-SARSA
    Reduced State SARSA Q-Learning
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
