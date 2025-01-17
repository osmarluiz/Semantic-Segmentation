import re
import torch.nn as nn

class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

class Metric(BaseObject):
    pass

class Loss(BaseObject):
    def __add__(self, other):
        if not isinstance(other, Loss):
            raise ValueError('Loss should be inherited from `Loss` class')
        return SumOfLosses(self, other)

    def __mul__(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError('Multiplier should be a number')
        return MultipliedLoss(self, value)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, value):
        return self.__mul__(value)

class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        name = f'{l1.__name__} + {l2.__name__}'
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def forward(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)

class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):
        name = f'{multiplier} * ({loss.__name__})' if '+' in loss.__name__ else f'{multiplier} * {loss.__name__}'
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def forward(self, *inputs):
        return self.multiplier * self.loss(*inputs)
