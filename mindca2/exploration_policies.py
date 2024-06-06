import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_action(self, temp, chs, qvals) -> tuple[int, int, float]:
        """

        :param temp: Temperature/epsilon
        :param chs: Channels to select from.
        :param qvals: q-value for each ch in chs.
        :param cell: Cell in which action is to be executed

        Returns (ch, idx) for selected channel where ch=chs[idx]
        """
        pass


class Greedy(Policy):
    """Greedy action selection. Select the action with the highest q-value"""

    def __init__(self, *args, **kwargs):
        pass

    def select_action(self, epsilon, chs, qvals):
        # Choose greedily
        idx = np.argmax(qvals)
        ch = chs[idx]
        return ch, idx

class Boltzmann(Policy):
    """
    A stochastic Boltzmann policy for selecting actions
    """

    def select_action(self, temp, chs, qvals, *args):
        max_idx = np.argmax(qvals)
        # Scale the q-values by subtracting the maximum q-value to avoid numerical
        # instabilities
        qvals_scaled = qvals - qvals[max_idx]
        exponentials = np.exp(qvals_scaled / temp)
        # Probability mass function
        probs = exponentials / np.sum(exponentials)
        # Select a channel indirectly by sampling among the channel indexes according to the probability
        # mass function
        idx = np.random.choice(range(len(chs)), p=probs)
        return chs[idx], idx
