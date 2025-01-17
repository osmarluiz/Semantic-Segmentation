import numpy as np

class Meter:
    """
    Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Resets the meter to default settings."""
        raise NotImplementedError

    def add(self, value):
        """
        Log a new value to the meter.
        Args:
            value: Next result to include.
        """
        raise NotImplementedError

    def value(self):
        """Get the value of the meter in the current state."""
        raise NotImplementedError


class AverageValueMeter(Meter):
    def __init__(self):
        super().__init__()
        self.reset()

    def add(self, value: float, n: int = 1):
        """
        Add a new value to the meter.
        Args:
            value: Next result to include.
            n: Number of occurrences of this value. Defaults to 1.
        """
        if n < 0:
            raise ValueError("n must be non-negative")

        self.sum += value * n
        self.n += n

        if self.n == 0:
            self.mean = np.nan
            self.std = np.nan
        elif self.n == 1:
            self.mean = value
            self.std = 0.0
        else:
            delta = value - self.mean
            self.mean += delta / self.n
            self.m_s += delta * (value - self.mean)
            self.std = np.sqrt(self.m_s / (self.n - 1))

    def value(self):
        """
        Return the current mean and standard deviation.
        Returns:
            Tuple containing the mean and standard deviation.
        """
        return self.mean, self.std

    def reset(self):
        """
        Resets the meter to its initial state.
        """
        self.n = 0
        self.sum = 0.0
        self.mean = np.nan
        self.m_s = 0.0
        self.std = np.nan
