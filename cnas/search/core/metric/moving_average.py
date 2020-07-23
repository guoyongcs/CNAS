import torch
import typing


class MovingAverageMetric(object):
    def __init__(self, gamma=0.9):
        self.n = 0
        self._value = 0.
        self.last = 0.
        self.gamma = gamma

    def reset(self) -> None:
        self.n = 0
        self._value = 0.
        self.last = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.last = value.item()
        elif isinstance(value, (int, float)):
            self.last = value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self.n += 1
        if self.n == 1:
            self._value = self.last
        else:
            self._value = self._value * self.gamma + self.last * (1-self.gamma)

    @property
    def value(self) -> float:
        if self.n == 0:
            raise ValueError("The container is empty.")
        return self._value
