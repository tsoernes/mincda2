import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_action(self, chs: list[int], *args, **kwargs) -> tuple[int, int]:
        """
        Select an action (i.e. a channel from the given channels)

        :param chs: Channels to select from.

        Returns (ch, idx) for selected channel where ch=chs[idx]
        """
        raise NotImplementedError


class Greedy(Policy):
    """Greedy action selection. Select the action with the highest q-value"""

    def __init__(self, *args, **kwargs):
        pass

    def select_action(
        self,
        chs: npt.NDArray[np.int_],
        qvals: npt.NDArray[np.float_],
        *args,
        **kwargs,
    ) -> tuple[int, int]:
        """
        Select an action (i.e. a channel from the given channels)

        :param chs: Channels to select from.
        :param qvals: q-value for each ch in chs.

        Returns (ch, idx) for selected channel where ch=chs[idx]
        """
        # Choose greedily
        idx = int(np.argmax(qvals))
        ch = chs[idx]
        return ch, idx


class Boltzmann(Policy):
    """
    A stochastic Boltzmann policy for selecting actions
    """

    def __init__(
        self, epsilon, epsilon_decay=0.9999, epsilon_log_decay: float | None = None
    ):
        """
        :param epsilon: Initial temperature/epsilon.
            A scaling factor that's usually decreased
        with time.
        """
        # Initial epsilon
        self.epsilon0 = epsilon
        # Current epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_log_decay = epsilon_log_decay

    def select_action(
        self,
        chs: npt.NDArray[np.int_],
        qvals: npt.NDArray[np.float_],
        time: float,
    ) -> tuple[int, int]:
        """
        Select an action (i.e. a channel from the given channels)

        :param chs: Channels to select from.
        :param qvals: q-value for each ch in chs.
        :param time: time of event

        Returns (ch, idx) for selected channel where ch=chs[idx]
        """
        # Scale the q-values by subtracting the maximum q-value to avoid numerical
        # instabilities
        qvals_scaled = qvals - qvals.max()
        exponentials = np.exp(qvals_scaled / self.epsilon)
        # Boltzmann Probability mass function
        probs = exponentials / np.sum(exponentials)
        # Select a channel indirectly by sampling among the channel indexes according to the probability
        # mass function
        idx: int = np.random.choice(range(len(chs)), p=probs)

        # Decay epsilon
        if self.epsilon_log_decay:
            self.epsilon = self.epsilon0 / np.sqrt(time * 60 / self.epsilon_log_decay)
        else:
            self.epsilon *= self.epsilon_decay
        return chs[idx], idx
