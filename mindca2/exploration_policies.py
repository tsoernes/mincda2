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
        self, chs: npt.NDArray[np._IntType], qvals: npt.NDArray[np._FloatType]
    ) -> tuple[np.int_, np.intp]:
        """
        Select an action (i.e. a channel from the given channels)

        :param chs: Channels to select from.
        :param qvals: q-value for each ch in chs.

        Returns (ch, idx) for selected channel where ch=chs[idx]
        """
        # Choose greedily
        idx = np.argmax(qvals)
        ch = chs[idx]
        return ch, idx


class Boltzmann(Policy):
    """
    A stochastic Boltzmann policy for selecting actions
    """

    def __init__(self, temp: float):
        """
        :param temp: Temperature/epsilon
        """
        self.temp = temp

    def select_action(
        self, chs: npt.NDArray[np._IntType], qvals: npt.NDArray[np._FloatType]
    ) -> tuple[np._IntType, int]:
        """
        Select an action (i.e. a channel from the given channels)

        :param chs: Channels to select from.
        :param qvals: q-value for each ch in chs.

        Returns (ch, idx) for selected channel where ch=chs[idx]
        """
        # Scale the q-values by subtracting the maximum q-value to avoid numerical
        # instabilities
        qvals_scaled = qvals - qvals.max()
        exponentials = np.exp(qvals_scaled / self.temp)
        # Boltzmann Probability mass function
        probs = exponentials / np.sum(exponentials)
        # Select a channel indirectly by sampling among the channel indexes according to the probability
        # mass function
        idx: int = np.random.choice(range(len(chs)), p=probs)
        return chs[idx], idx
