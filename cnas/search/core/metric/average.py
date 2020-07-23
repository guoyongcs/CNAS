import torch
import typing


class AverageMetric(object):
    def __init__(self):
        self.n = 0
        self._value = 0.
        self.last = 0.

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
        self._value += self.last

    @property
    def value(self) -> float:
        if self.n == 0:
            raise ValueError("The container is empty.")
        return self._value / self.n
