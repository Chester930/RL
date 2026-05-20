"""
Epsilon schedulers for exploration decay.

Usage:
    # Linear: start=1.0, end=0.01, decay over 100k steps
    sched = LinearSchedule(start=1.0, end=0.01, total_steps=100_000)
    eps = sched.get(step=50_000)   # -> 0.505

    # Exponential: start=1.0, end=0.01, half-life every 10k steps
    sched = ExponentialSchedule(start=1.0, end=0.01, decay=0.9999)
    eps = sched.get(step=10_000)
"""


class LinearSchedule:
    """
    Linearly decay epsilon from `start` to `end` over `total_steps`.

    After `total_steps` the value remains at `end`.

    Args:
        start:       Initial epsilon value (e.g. 1.0).
        end:         Final (minimum) epsilon value (e.g. 0.01).
        total_steps: Number of steps over which to decay.
        warmup:      Number of steps to keep epsilon at `start` before decaying.
    """

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.01,
        total_steps: int = 100_000,
        warmup: int = 0,
    ):
        assert start >= end >= 0.0, "start must be >= end >= 0"
        assert total_steps > 0
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.warmup = warmup

    def get(self, step: int) -> float:
        """Return the current epsilon value at the given training step."""
        if step < self.warmup:
            return self.start
        decay_step = step - self.warmup
        decay_range = max(1, self.total_steps - self.warmup)
        frac = min(1.0, decay_step / decay_range)
        return self.start + frac * (self.end - self.start)

    def __repr__(self) -> str:
        return (
            f"LinearSchedule(start={self.start}, end={self.end}, "
            f"total_steps={self.total_steps}, warmup={self.warmup})"
        )


class ExponentialSchedule:
    """
    Exponential epsilon decay: epsilon = max(end, start * decay^step).

    Args:
        start: Initial epsilon (e.g. 1.0).
        end:   Minimum epsilon (floor).
        decay: Multiplicative factor per step (e.g. 0.9999).
    """

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.01,
        decay: float = 0.9999,
    ):
        assert 0.0 < decay < 1.0, "decay must be in (0, 1)"
        assert start >= end >= 0.0
        self.start = start
        self.end = end
        self.decay = decay

    def get(self, step: int) -> float:
        """Return the current epsilon value at the given training step."""
        return max(self.end, self.start * (self.decay ** step))

    def __repr__(self) -> str:
        return (
            f"ExponentialSchedule(start={self.start}, end={self.end}, "
            f"decay={self.decay})"
        )


class CosineSchedule:
    """
    Cosine annealing schedule (commonly used for learning rates).

    Decays value from `start` to `end` following a cosine curve,
    which provides a smooth start and slow end compared to linear.
    """

    import math as _math

    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def get(self, step: int) -> float:
        import math
        frac = min(1.0, step / max(1, self.total_steps))
        cosine = (1 - math.cos(math.pi * frac)) / 2
        return self.start + cosine * (self.end - self.start)
