import numpy
import torch
import typing


class Statistics(object):
    def __init__(self):
        self._values = []
        self.last = 0.

    def reset(self) -> None:
        self._values = []
        self.last = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.last = value.item()
        elif isinstance(value, (int, float)):
            self.last = value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self._values.append(self.last)

    @property
    def avg(self) -> float:
        if not self._values:
            raise ValueError("The container is empty.")
        return float(numpy.array(self._values).mean())

    @property
    def std(self) -> float:
        if not self._values:
            raise ValueError("The container is empty.")
        return float(numpy.array(self._values).std())
